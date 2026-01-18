from typing import Dict, List, Set, Iterable, Sequence, Optional, Deque
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupportBase
import logging
import itertools
from copy import copy
from collections import deque
def merge_single_node(node: Node, id: Optional[int]):
    if node in assignment:
        partitions_by_id[assignment[node]].remove_node(node)
    if id is None:
        assignment.pop(node)
    elif id not in partitions_by_id:
        assignment[node] = id
        partitions_by_id[id] = Partition(id=id, nodes=[node])
    else:
        assignment[node] = id
        partitions_by_id[id].add_node(node)