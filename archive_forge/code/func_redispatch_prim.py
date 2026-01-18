import inspect
import warnings
from functools import wraps
from itertools import chain
from typing import Callable, NamedTuple, Optional, overload, Sequence, Tuple
import torch
import torch._prims_common as utils
from torch._prims_common import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def redispatch_prim(args, kwargs):
    with torch._C._AutoDispatchBelowAutograd():
        old = torch._C._dispatch_tls_is_dispatch_key_excluded(torch._C.DispatchKey.ADInplaceOrView)
        return prim(*args, **kwargs)