import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def size_based_partition(self) -> None:
    """This method is to partition the fx module based on memory size.
        It uses greedy approach. The result may not be the best.
        The basic idea is:
        Step 1:
        Find a device which has enough memory to fit the current node, create a empty partition
        with the size of that device.
        Then keep adding the following nodes into the partition until the partition is full.
        Step 2:
        Repeat Step 1 until no device left
        Step 3:
        If some nodes are left, create a partition for each left node (single node partition).
        and then try to map those partitions into logical devices with enough mem left.
        """

    def find_device_based_on_size(node) -> Device:
        """Given a node, this function is to find a logical device
            that could fit the node.
            """
        mem_size_needed = get_extra_size_of(node, set())
        device = Device('', -1, -1)
        for d in self.devices:
            if d not in occupied_devices and d.available_mem_bytes >= mem_size_needed:
                device = d
                break
        if device.available_mem_bytes < 0:
            raise RuntimeError(str(node) + 'is too large to fit any device')
        occupied_devices.append(device)
        return device
    partition_to_left_mem_bytes: Dict[Partition, int] = {}
    occupied_devices: List[Device] = []
    partition = self.create_partition()
    for node in self.graph_module.graph.nodes:
        if node.op in {'call_module', 'call_method', 'call_function'}:
            if len(self.partitions) <= len(self.devices):
                total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                if partition.used_mem_bytes == 0:
                    device = find_device_based_on_size(node)
                    occupied_devices.append(device)
                    partition_to_left_mem_bytes[partition] = device.available_mem_bytes
                    partition.logical_device_ids.append(device.logical_id)
                elif partition_to_left_mem_bytes[partition] < total_size_of_input_nodes:
                    if len(self.partitions) == len(self.devices):
                        non_single_node_partitions = self.partitions[:]
                        self.create_single_node_partition(node)
                        continue
                    device = find_device_based_on_size(node)
                    partition = self.create_partition()
                    total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                    partition_to_left_mem_bytes[partition] = device.available_mem_bytes
                    partition.logical_device_ids.append(device.logical_id)
                partition.add_node(node)
                partition_to_left_mem_bytes[partition] -= total_size_of_input_nodes
            else:
                self.create_single_node_partition(node)
    reorganize_partitions(self.partitions)
    self.node_to_partition = get_node_to_partition_mapping(self.partitions)
    found_partition_to_device_mapping = get_device_to_partitions_mapping(self.partitions, self.devices)
    if not found_partition_to_device_mapping:
        raise RuntimeError('Cannot Get a Valid Partition to Logical Device Mapping')
    return