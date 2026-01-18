from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
import torch.nn.functional as F
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
from torch.testing._internal.common_nn import module_tests, new_module_tests
from torch.testing._internal.common_utils import is_iterable_of_tensors
import collections
from copy import deepcopy
from typing import Any, Dict, List, Union
import math  # noqa: F401
from torch import inf
def create_input(call_args, requires_grad=True, non_contiguous=False, call_kwargs=None, dtype=torch.float, device=None):
    if not isinstance(call_args, tuple):
        call_args = (call_args,)

    def map_arg(arg):

        def maybe_non_contig(tensor):
            if not non_contiguous or tensor.numel() < 2:
                return tensor.clone()
            return noncontiguous_like(tensor)

        def conjugate(tensor):
            return tensor.conj()
        if isinstance(arg, (torch.Size, dont_convert)):
            return arg
        elif isinstance(arg, tuple) and len(arg) == 0:
            var = conjugate(torch.randn((), dtype=dtype, device=device))
            var.requires_grad = requires_grad
            return var
        elif isinstance(arg, tuple) and (not isinstance(arg[0], torch.Tensor)):
            return conjugate(maybe_non_contig(torch.randn(*arg, dtype=dtype, device=device))).requires_grad_(requires_grad)
        elif isinstance(arg, non_differentiable):
            if isinstance(arg.tensor, torch.Tensor):
                return conjugate(maybe_non_contig(arg.tensor.to(device=device)))
            return conjugate(maybe_non_contig(arg.tensor.to(device=device)))
        elif isinstance(arg, torch.Tensor):
            if arg.is_complex() != dtype.is_complex:
                raise RuntimeError('User provided tensor is real for a test that runs with complex dtype, ', 'which is not supported for now')
            v = conjugate(maybe_non_contig(arg)).detach().to(device=device).clone()
            v.requires_grad = requires_grad and (v.is_floating_point() or v.is_complex())
            return v
        elif callable(arg):
            return map_arg(arg(dtype=dtype, device=device))
        else:
            return arg
    args_out = tuple((map_arg(arg) for arg in call_args))
    kwargs_out = {k: map_arg(v) for k, v in call_kwargs.items()} if call_kwargs else {}
    return (args_out, kwargs_out)