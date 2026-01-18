import copy
from typing import (
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def make_functional_with_buffers(model: nn.Module, disable_autograd_tracking: bool=False) -> Tuple[FunctionalModuleWithBuffers, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    """make_functional_with_buffers(model, disable_autograd_tracking=False) -> func, params, buffers

    Given a ``torch.nn.Module``, make_functional_with_buffers extracts the
    state (params and buffers) and returns a functional version of the model
    ``func`` that can be invoked like a function.

    ``func`` can be invoked as follows:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional_with_buffers

        x = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params, buffers = make_functional_with_buffers(model)
        func(params, buffers, x)

    And here is an example of applying the grad transform over the parameters
    of a model:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional_with_buffers, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params, buffers = make_functional_with_buffers(model)

        def compute_loss(params, buffers, x, t):
            y = func(params, buffers, x)
            return nn.functional.mse_loss(y, t)

        grad_weights = grad(compute_loss)(params, buffers, x, t)

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
    return FunctionalModuleWithBuffers._create_from(model, disable_autograd_tracking=disable_autograd_tracking)