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
def get_suffix_str(number):
    index = max(0, min(len(suffixes) - 1, (len(str(number)) - 3) // 3))
    return suffixes[index]