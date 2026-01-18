import sys
import torch
from torch.fx.graph import (
from torch.ao.quantization.utils import Pattern
from .quantize_handler import (
from ..qconfig import (
from ..utils import (
from .graph_module import (
from torch.nn.utils.parametrize import type_before_parametrizations
from typing import Any, Dict, List, Callable, Optional, Tuple, Type, Set, Iterable
def record_match(pattern, node, last_node, matched_node_pattern, match_map):
    if isinstance(pattern, tuple):
        s, *args = pattern
        is_single_arg = len(args) == 1
        current_node_pattern: List[Node] = []
        record_match(s, node, last_node, matched_node_pattern, match_map)
        if pattern[0] is not getattr:
            for subpattern, arg in zip(args, node.args):
                record_match(subpattern, arg, node, current_node_pattern, match_map)
        if len(current_node_pattern) > 1:
            if is_single_arg:
                matched_node_pattern.append(tuple(current_node_pattern))
            else:
                matched_node_pattern.extend(list(current_node_pattern))
        else:
            matched_node_pattern.append(current_node_pattern[0])
    else:
        matched_node_pattern.append(node)