import abc
import contextlib
import weakref
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
def get_gradient_edge(tensor):
    """Get the gradient edge for computing the gradient of the given Tensor.

    In particular, it is equivalent to call
    ``g = autograd.grad(loss, input)`` and ``g = autograd.grad(loss, get_gradient_edge(input))``.
    """
    if not tensor.requires_grad:
        raise RuntimeError('It is not possible to get the gradient edge for a Tensor that does not require gradients')
    grad_fn = tensor.grad_fn
    if grad_fn is None:
        grad_fn = tensor.view_as(tensor).grad_fn.next_functions[0][0]
    return GradientEdge(grad_fn, tensor.output_nr)