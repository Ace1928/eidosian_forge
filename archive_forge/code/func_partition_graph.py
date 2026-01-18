import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def partition_graph(self, fx_module: GraphModule, torch_module: torch.nn.Module, partitioner_config: PartitionerConfig) -> PartitionResult:
    """Given the fx module, torch module and partitioner_config,
        find the partitions, do the partitions,
        and then return a DAG and a new fx module with submodule nodes (partitions)
        """
    self.graph_module = fx_module
    self.torch_module = torch_module
    self.devices = partitioner_config.devices
    if len(self.devices) == 0:
        raise RuntimeError('No devices')
    get_size_of_all_nodes(self.graph_module)
    nodes = self.graph_module.graph.nodes
    if all((node.op in {'placeholder', 'get_attr', 'output'} for node in nodes)):
        raise RuntimeError('No Partition since no operations in the module')
    total_size_of_graph = 0
    for node in nodes:
        if node.op == 'output':
            break
        total_size_of_graph += node.size_bytes.total_size
    device_with_max_mem = max(self.devices, key=lambda d: d.available_mem_bytes)
    if partitioner_config.mode == PartitionMode.aot_based:
        self.aot_based_partition(partitioner_config.node_to_partition_mapping, partitioner_config.partition_to_logical_device_mapping)
    elif total_size_of_graph <= device_with_max_mem.available_mem_bytes:
        self.find_single_partition(total_size_of_graph, logical_device_id=device_with_max_mem.logical_id)
    elif total_size_of_graph > sum([d.available_mem_bytes for d in self.devices]):
        raise RuntimeError('Devices have no enough memory for the module')
    elif partitioner_config.mode == PartitionMode.sparse_nn:
        available_mem_bytes = self.devices[0].available_mem_bytes
        if not all((device.available_mem_bytes == available_mem_bytes for device in self.devices)):
            raise RuntimeError('All devices must have same memory size!')
        self.sparse_nn_partition(available_mem_bytes)
    elif partitioner_config.mode == PartitionMode.cost_aware:
        self.cost_aware_partition(partitioner_config.transfer_rate_bytes_per_sec, partitioner_config.node_to_latency_mapping)
    elif partitioner_config.mode == PartitionMode.kl_based:
        self.kl_based_partition(partitioner_config.transfer_rate_bytes_per_sec, partitioner_config.node_to_latency_mapping)
    else:
        self.size_based_partition()
    if partitioner_config.saturate_host:
        self.saturate_host()
    module_with_submodules = self.do_partition()
    dag = self.dump_dag(module_with_submodules)
    ret = PartitionResult(dag, module_with_submodules)
    return ret