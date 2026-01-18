import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from typing import List, Any, Dict, Optional, Union, NamedTuple
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from torch._decomp import register_decomposition
from math import prod
from functools import wraps
def register_flop_formula(targets, get_raw=False):

    def register_fun(flop_formula):
        if not get_raw:
            flop_formula = shape_wrapper(flop_formula)
        register_decomposition(targets, registry=flop_registry, unsafe=True)(flop_formula)
        return flop_formula
    return register_fun