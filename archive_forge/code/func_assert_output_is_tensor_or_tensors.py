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
def assert_output_is_tensor_or_tensors(output: Any, api: str) -> None:
    if isinstance(output, torch.Tensor):
        return
    if not isinstance(output, tuple):
        raise RuntimeError(f'{api}: Expected output of f to be a Tensor or Tensors, got {type(output)}')
    if len(output) == 0:
        raise RuntimeError(f'{api}: Expected output of f to be a non-empty tuple of Tensors.')
    for out in output:
        if isinstance(out, torch.Tensor):
            continue
        raise RuntimeError(f'{api}: Expected output of f to be a Tensor or Tensors, got {type(out)} as an output')