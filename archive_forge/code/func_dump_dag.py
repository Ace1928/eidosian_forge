import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def dump_dag(self, module_with_submodules: GraphModule) -> DAG:
    """Return the dag structure and the new fx module with submodules."""
    dag = DAG()
    for node in module_with_submodules.graph.nodes:
        if node.op == 'output':
            break
        if node.op in {'placeholder', 'get_attr'}:
            continue
        if node.target == operator.__getitem__:
            continue
        input_nodes: Dict[Node, None] = {}
        map_arg(node.args, input_nodes.setdefault)
        map_arg(node.kwargs, input_nodes.setdefault)
        if len(node.users) > 1:
            output_nodes = list(node.users)
        else:
            output_nodes = [node]
        partition_id = int(node.name.rsplit('_', 1)[-1])
        device_ids = self.partitions[partition_id].logical_device_ids
        size_bytes = self.partitions[partition_id].used_mem_bytes
        dag.create_node(node, list(input_nodes), output_nodes, device_ids, size_bytes)
    return dag