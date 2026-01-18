from typing import Callable, Union, Tuple, List, Any, Optional
import torch
from functools import partial, wraps
import contextlib
from torch.utils._pytree import (
from torch.utils import _pytree as pytree
from torch.fx.experimental import const_fold
from torch.fx.experimental.proxy_tensor import make_fx
import torch.autograd.forward_ad as fwAD
from torch._subclasses.functional_tensor import FunctionalTensor
from .vmap import doesnt_support_saved_tensors_hooks, get_chunk_sizes
from .apis import vmap
from torch._C._functorch import (
from torch._functorch.utils import exposed_in, argnums_t
def assert_flat_tuple_of_tensors(elts: Any, api: str, argname: str) -> None:
    if not isinstance(elts, tuple):
        raise RuntimeError(f'{api}: Expected {argname} to be a tuple of Tensors, got {type(elts)}')
    for elt in elts:
        if isinstance(elt, torch.Tensor):
            continue
        raise RuntimeError(f'{api}: Expected {argname} to be a tuple of Tensors, got a tuple with an element of type {type(elt)}')
    if len(elts) == 0:
        raise RuntimeError(f'{api}: Expected {argname} to be a non-empty tuple of Tensors.')