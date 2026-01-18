import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
def get_numerical_jacobian(fn, inputs, target=None, eps=0.001, grad_out=1.0):
    """Compute the numerical Jacobian for a given fn and its inputs.

    This is a Deprecated API.

    Args:
        fn: the function to compute the Jacobian for (must take inputs as a tuple)
        input: input to `fn`
        target: the Tensors wrt whom Jacobians are calculated (default=`input`)
        eps: the magnitude of the perturbation during finite differencing
             (default=`1e-3`)

    Returns:
        A list of Jacobians of `fn` (restricted to its first output) with respect to
        each input or target, if provided.

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    warnings.warn("get_numerical_jacobian was part of PyTorch's private API and not meant to be exposed. We are deprecating it and it will be removed in a future version of PyTorch. If you have a specific use for this or feature request for this to be a stable API, please file us an issue at https://github.com/pytorch/pytorch/issues/new")
    if grad_out != 1.0:
        raise ValueError('Expected grad_out to be 1.0. get_numerical_jacobian no longer supports values of grad_out != 1.0.')

    def fn_pack_inps(*inps):
        return fn(inps)
    jacobians = _get_numerical_jacobian(fn_pack_inps, inputs, None, target, eps)
    return tuple((jacobian_for_each_output[0] for jacobian_for_each_output in jacobians))