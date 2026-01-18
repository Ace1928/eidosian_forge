import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def aot_based_partition(self, node_to_partition_mapping, partition_to_logical_device_mapping):
    """This function helps to rebuild the partitions given the nodes and its
        corresponding partition id
        """
    partition_id_to_partition_mapping: Dict[int, Partition] = {}
    self.node_to_partition = node_to_partition_mapping
    for node in self.node_to_partition:
        partition_id = self.node_to_partition[node]
        if partition_id not in partition_id_to_partition_mapping:
            partition = Partition(partition_id)
            self.partitions.append(partition)
            partition_id_to_partition_mapping[partition_id] = partition
            partition.logical_device_ids = partition_to_logical_device_mapping[partition_id]
        else:
            partition = partition_id_to_partition_mapping[self.node_to_partition[node]]
        partition.add_node(node)