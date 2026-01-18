from contextlib import contextmanager
import torch
import functools
from torch._decomp import decomposition_table
from typing import Callable, Dict
from torch.utils._pytree import tree_map_only
@contextmanager
def allow_smaller_batches(args, kwargs):

    def allow(ew):
        ew.set_allow_smaller_batches(True)

    def reset(ew):
        ew.set_allow_smaller_batches(False)
    tree_map_only(ExpandedWeight, allow, args)
    tree_map_only(ExpandedWeight, allow, kwargs)
    try:
        yield
    finally:
        tree_map_only(ExpandedWeight, reset, args)
        tree_map_only(ExpandedWeight, reset, kwargs)