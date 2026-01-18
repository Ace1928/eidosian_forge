from typing import Dict, List, Set, Iterable, Sequence, Optional, Deque
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupportBase
import logging
import itertools
from copy import copy
from collections import deque
def partition_and_fuse(self) -> GraphModule:
    partitions = self.propose_partitions()
    fused_gm = self.fuse_partitions(partitions)
    return fused_gm