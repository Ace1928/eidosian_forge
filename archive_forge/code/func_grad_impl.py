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
def grad_impl(func: Callable, argnums: argnums_t, has_aux: bool, args, kwargs):
    func = lazy_dynamo_disable(func)
    results = grad_and_value(func, argnums, has_aux=has_aux)(*args, **kwargs)
    if has_aux:
        grad, (_, aux) = results
        return (grad, aux)
    grad, _ = results
    return grad