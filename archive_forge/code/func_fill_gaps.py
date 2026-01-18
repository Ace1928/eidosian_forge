import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def fill_gaps(self, ranges: List[_Range], min_: int, max_: int) -> List[_Range]:
    cur = min_
    filled_ranges = []
    for a, b in ranges:
        if cur < a:
            filled_ranges.append((cur, a))
        filled_ranges.append((a, b))
        cur = b
    if filled_ranges[-1][1] < max_:
        filled_ranges.append((filled_ranges[-1][1], max_))
    return filled_ranges