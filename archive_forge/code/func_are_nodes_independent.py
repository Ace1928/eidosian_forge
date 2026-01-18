import torch
from torch.fx.node import Node
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.passes.tools_common import legalize_graph
import itertools
import operator
from typing import Dict, List, Tuple
def are_nodes_independent(nodes: List[Node]):
    """
    Check if all of the given nodes are pairwise-data independent.

    Arguments:
        nodes: The nodes to check for data dependencies.

    Returns:
        True if any pair in nodes has a data dependency.
    """
    for i, j in itertools.combinations(nodes, 2):
        if may_depend_on(i, j) or may_depend_on(j, i):
            return False
    return True