import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def merge_consecutive_inputs(self, inputs: List[Union[torch.fx.Node, int]]) -> List[Union[torch.fx.Node, _Range]]:
    """
        Merge consecutive inputs going into a user node.

        For e.g.
        [arg0, 0, 1, 2, arg1] -> [arg0, (0, 2), arg1]
        """
    merged_ranges = []
    cur_range = None
    for input_ in inputs:
        if isinstance(input_, int):
            if not cur_range:
                cur_range = [input_, input_]
            elif input_ == cur_range[1] + 1:
                cur_range[1] += 1
            else:
                merged_ranges.append(tuple(cur_range))
                cur_range = [input_, input_]
        else:
            if cur_range:
                merged_ranges.append(tuple(cur_range))
                cur_range = None
            merged_ranges.append(input_)
    if cur_range:
        merged_ranges.append(tuple(cur_range))
    return merged_ranges