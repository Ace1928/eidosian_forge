import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def get_device_partition_stats(partitions: List[Partition], devices: List[Device]) -> Tuple[Dict[Device, List[Partition]], Dict[Device, int], List[Partition]]:
    """Given a list of partitions and a list of devices, returns:
    1. A mapping from device to partitions on it;
    2. A mapping from device to its remaining memory size;
    3. A list of partitions that do not have a device.
    """
    logical_id_to_device = get_logical_id_to_device(devices)
    device_to_partitions: Dict[Device, List[Partition]] = {}
    device_to_left_mem_bytes: Dict[Device, int] = {}
    for d in devices:
        device_to_partitions[d] = []
        device_to_left_mem_bytes[d] = d.available_mem_bytes
    no_device_partitions = []
    for partition in partitions:
        if partition.logical_device_ids != []:
            for logical_id in partition.logical_device_ids:
                device = logical_id_to_device[logical_id]
                device_to_partitions[device].append(partition)
                device_to_left_mem_bytes[device] -= partition.used_mem_bytes
        else:
            no_device_partitions.append(partition)
    return (device_to_partitions, device_to_left_mem_bytes, no_device_partitions)