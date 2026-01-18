import os
from collections import namedtuple
from typing import Any
import torch
from .grad_mode import _DecoratorContextManager
def make_dual(tensor, tangent, *, level=None):
    """Associate a tensor value with its tangent to create a "dual tensor" for forward AD gradient computation.

    The result is a new tensor aliased to :attr:`tensor` with :attr:`tangent` embedded
    as an attribute as-is if it has the same storage layout or copied otherwise.
    The tangent attribute can be recovered with :func:`unpack_dual`.

    This function is backward differentiable.

    Given a function `f` whose jacobian is `J`, it allows one to compute the Jacobian-vector product (`jvp`)
    between `J` and a given vector `v` as follows.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> with dual_level():
        ...     inp = make_dual(x, v)
        ...     out = f(inp)
        ...     y, jvp = unpack_dual(out)

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.

    """
    if os.environ.get('PYTORCH_JIT', '1') == '1' and __debug__:
        from torch._decomp import decompositions_for_jvp
    if level is None:
        level = _current_level
    if level < 0:
        raise RuntimeError('Trying to create a dual Tensor for forward AD but no level exists, make sure to enter_dual_level() first.')
    if not (tensor.is_floating_point() or tensor.is_complex()):
        raise ValueError(f'Expected primal to be floating point or complex, but got: {tensor.dtype}')
    if not (tangent.is_floating_point() or tangent.is_complex()):
        raise ValueError(f'Expected tangent to be floating point or complex, but got: {tangent.dtype}')
    return torch._VF._make_dual(tensor, tangent, level=level)