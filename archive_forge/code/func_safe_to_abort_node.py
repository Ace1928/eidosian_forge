import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def safe_to_abort_node(node: torch.fx.Node):
    """
    1. the input nodes of the node should come from the same parent
    2. the user of all the input nodes should be only one
    """
    prev_node = None
    for arg in node.args[0]:
        if len(arg.users) != 1 or arg.target != operator.getitem:
            return False
        if prev_node is None:
            prev_node = arg.args[0]
        elif arg.args[0] != prev_node:
            return False
    return True