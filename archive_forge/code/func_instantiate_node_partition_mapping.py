import inspect
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING
from collections import OrderedDict
import logging
import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
def instantiate_node_partition_mapping(node):
    partition_name = str(split_callback(node))
    partition = partitions.get(partition_name)
    if partition is None:
        partitions[partition_name] = partition = Partition(partition_name)
    partition.node_names.append(node.name)
    node._fx_partition = partition_name