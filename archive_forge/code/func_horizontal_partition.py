import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sympy import Integer
from .. import metrics
from ..scheduler import SchedulerNode
from ..utils import ceildiv, Placeholder
from ..virtualized import V
from .common import IndentedBuffer, Kernel
from .triton import TritonKernel
from .triton_utils import config_of, signature_to_meta
@staticmethod
def horizontal_partition(subkernel_nodes, triton_scheduling):
    """Generates a list of lists of node info tuples which consist of (fused_nodes, tiling, numel, rnumel)
        for each subkernel node where each sublist is guaranteed to not exceed CUDA limits for number of args
        (read/writes) and to have the same 2D or 1D blocking strategy."""
    assert len(subkernel_nodes) >= 1
    partition_state_1d = PartitionState([], [], 0)
    yelem_to_partition_state_2d: Dict[Integer, PartitionState] = defaultdict(lambda: PartitionState([], [], 0))
    for node in subkernel_nodes:
        fused_nodes = node.get_nodes()
        _, (numel, rnumel) = max(fused_nodes, key=lambda x: int(x.is_reduction())).group
        tiled_groups = triton_scheduling.select_tiling(fused_nodes, numel, rnumel)
        node_info = (fused_nodes, tiled_groups, numel, rnumel)
        read_writes = node.read_writes
        read_write_count = len(read_writes.reads) + len(read_writes.writes)
        if tiled_groups[1] == 1:
            ForeachKernel._update_partition(partition_state_1d, read_write_count, node_info)
        else:
            y_elem = tiled_groups[0]
            partition_state_2d = yelem_to_partition_state_2d[y_elem]
            ForeachKernel._update_partition(partition_state_2d, read_write_count, node_info)
    partition_state_1d.finalize()
    all_partitions = partition_state_1d.partitions
    for partition_state_2d in yelem_to_partition_state_2d.values():
        partition_state_2d.finalize()
        all_partitions.extend(partition_state_2d.partitions)
    return all_partitions