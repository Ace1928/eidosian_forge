import abc
import contextlib
import weakref
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
@abc.abstractmethod
def register_prehook(self, fn: Callable[..., Any]) -> RemovableHandle:
    """Register a backward pre-hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None

        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_outputs`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        Example::

            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> handle = b.grad_fn.register_prehook(lambda gI: (gI[0] * 2,))
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([2., 2., 2.])
            >>> handle.remove()
            >>> a.grad = None
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([1., 1., 1.])
        """
    ...