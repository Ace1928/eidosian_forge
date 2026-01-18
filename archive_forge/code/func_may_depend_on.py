import torch
from torch.fx.node import Node
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.passes.tools_common import legalize_graph
import itertools
import operator
from typing import Dict, List, Tuple
def may_depend_on(a: Node, b: Node, search_depth: int=6):
    """
    Determine if one node depends on another in a torch.fx.Graph.

    Arguments:
        a: The node that may have a dependency on b.
        b: The node that a may have a dependency on.
        search_depth: In the case of an indirect dependency, this function
                        searches upto this many nodes away in search of a
                        data dependency. If none is found, the function
                        makes the conservative assumption that there is a
                        dependency.

    Returns:
        True if a may depend on b, False if it definitely does not.
    """
    if a == b:
        return True
    if len(a.all_input_nodes) == 0:
        return False
    if search_depth == 0:
        return True
    for inp in a.all_input_nodes:
        if may_depend_on(inp, b, search_depth - 1):
            return True
    return False