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
def safe_unpack_dual(dual, strict):
    if not isinstance(dual, torch.Tensor):
        raise RuntimeError(f'{jvp_str}: expected f(*args) to return only tensors, got unsupported type {type(dual)}')
    primal, tangent = fwAD.unpack_dual(dual)
    if tangent is None:
        if strict:
            raise RuntimeError('jvp(f, primals, tangents, strict=True): The output of f is independent of the inputs. This is not allowed with strict=True.')
        tangent = torch.zeros_like(primal)
    return (primal, tangent)