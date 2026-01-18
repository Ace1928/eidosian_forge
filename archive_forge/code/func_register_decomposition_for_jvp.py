import inspect
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch._decomp
from torch import Tensor
from torch._prims_common.wrappers import _maybe_remove_out_wrapper
def register_decomposition_for_jvp(fn):
    return register_decomposition(fn, registry=decomposition_table_for_jvp)