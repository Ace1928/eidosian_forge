import copy
from typing import (
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def make_functional(model: nn.Module, disable_autograd_tracking: bool=False) -> Tuple[FunctionalModule, Tuple[Tensor, ...]]:
    """make_functional(model, disable_autograd_tracking=False) -> func, params

    Given a ``torch.nn.Module``, :func:`make_functional` extracts the state
    (params) and returns a functional version of the model, ``func``. This
    makes it so that it is possible use transforms over the parameters of
    ``model``.

    ``func`` can be invoked as follows:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional

        x = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)
        func(params, x)

    And here is an example of applying the grad transform over the parameters
    of a model.

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)

        def compute_loss(params, x, t):
            y = func(params, x)
            return nn.functional.mse_loss(y, t)

        grad_weights = grad(compute_loss)(params, x, t)

    If the model has any buffers, please use :func:`make_functional_with_buffers` instead.

    Args:
        model (torch.nn.Module): Input model.
        disable_autograd_tracking (bool): Flag to disable gradients tracking for output parameters.
            The returned params are unrelated to the set of params from the original model. If False (default),
            the params will have ``requires_grad=True`` on them (aka they will be trackable with regular
            PyTorch autograd), matching the requires_grad-ness of the params from the original model.
            Otherwise, the returned params will have ``requires_grad=False``. Default, False.
            If you plan on using regular PyTorch autograd (e.g., if you want to call ``.backward()`` or
            ``torch.autograd.grad()``, then set ``disable_autograd_tracking=False``.
            Otherwise, if you're only planning on using functorch's gradient transforms,
            then please set ``disable_autograd_tracking=True`` to avoid unnecessarily tracking
            history with PyTorch autograd.

    """
    buffers = list(model.buffers())
    if len(buffers) > 0:
        raise RuntimeError('make_functional(model): `model` has buffers. Please use make_functional_with_buffers(model) instead.')
    return FunctionalModule._create_from(model, disable_autograd_tracking=disable_autograd_tracking)