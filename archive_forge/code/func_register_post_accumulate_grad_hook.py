import copyreg
import enum
import functools
import warnings
from collections import OrderedDict
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._namedtensor_internals import (
from torch.overrides import (
from torch.utils.dlpack import DLDeviceType
def register_post_accumulate_grad_hook(self, hook):
    """Registers a backward hook that runs after grad accumulation.

        The hook will be called after all gradients for a tensor have been accumulated,
        meaning that the .grad field has been updated on that tensor. The post
        accumulate grad hook is ONLY applicable for leaf tensors (tensors without a
        .grad_fn field). Registering this hook on a non-leaf tensor will error!

        The hook should have the following signature::

            hook(param: Tensor) -> None

        Note that, unlike other autograd hooks, this hook operates on the tensor
        that requires grad and not the grad itself. The hook can in-place modify
        and access its Tensor argument, including its .grad field.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks. Since
            this hook runs during the backward pass, it will run in no_grad mode (unless
            create_graph is True). You can use torch.enable_grad() to re-enable autograd
            within the hook if you need it.

        Example::

            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> lr = 0.01
            >>> # simulate a simple SGD update
            >>> h = v.register_post_accumulate_grad_hook(lambda p: p.add_(p.grad, alpha=-lr))
            >>> v.backward(torch.tensor([1., 2., 3.]))
            >>> v
            tensor([-0.0100, -0.0200, -0.0300], requires_grad=True)

            >>> h.remove()  # removes the hook
        """
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.register_post_accumulate_grad_hook, (self,), self, hook)
    if not self.requires_grad:
        raise RuntimeError("cannot register a hook on a tensor that doesn't require gradient")
    if self.grad_fn is not None:
        raise RuntimeError('post accumulate grad hooks cannot be registered on non-leaf tensors')
    if self._post_accumulate_grad_hooks is None:
        self._post_accumulate_grad_hooks: Dict[Any, Any] = OrderedDict()
    handle = hooks.RemovableHandle(self._post_accumulate_grad_hooks)
    self._post_accumulate_grad_hooks[handle.id] = hook
    return handle