import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def create_partition(self) -> Partition:
    """Create a partition and append it to self.partitions."""
    partition_id = len(self.partitions)
    partition = Partition(partition_id)
    self.partitions.append(partition)
    return partition