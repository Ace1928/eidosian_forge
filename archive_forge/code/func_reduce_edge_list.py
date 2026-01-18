import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from .tensor_utils import tensor_tree_map, tree_map
def reduce_edge_list(l: List[bool]) -> None:
    tally = True
    for i in range(len(l)):
        reversed_idx = -1 * (i + 1)
        l[reversed_idx] &= tally
        tally = l[reversed_idx]