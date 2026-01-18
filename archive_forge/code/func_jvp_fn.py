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
def jvp_fn(*tangents):
    flat_tangents, tangent_argspec = tree_flatten(tangents)
    if tangent_argspec != primals_argspec:
        raise RuntimeError(f'Expected the tangents {tangent_argspec} to have the same argspec as the primals {primals_argspec}')
    forward_ad_checks(flat_tangents)
    flat_output = const_folded_jvp_graph(*flat_tangents)
    return tree_unflatten(flat_output, output_spec)