import contextlib
from typing import Any, Callable, Dict, Generator, Optional, Set, Tuple, Type, cast
import torch.nn as nn
def auto_wrap(module: nn.Module, auto_wrap_policy: Optional[Callable]=None, **kwargs: Any) -> nn.Module:
    """
    Annotate that a module should be wrapped with the *wrapper_cls* from the
    :func:`enable_wrap` context (if the context exists) and recursively wrap
    children modules that meet the criteria given by :func:`auto_wrap_policy`. This
    is useful for wrapping large complex layers.

    .. note:: auto_wrap can only be applied to a module once because it
        assumes none of the sub-modules is already wrapped and uses that
        assumption to compute the wrapped vs. unwrapped parameters.
        To get around this limitation, users can pre-assign ``wrapper_config``
        attributes to the sub-modules they want to wrap (in multiple passes)
        and then uses the ``config_auto_wrap_policy``.

    .. warning:: It is not recommended to use :func:`auto_wrap` with
        :class:`FullyShardedDataParallel` on modules that have shared
        parameters, as the parameter sharing may be broken (i.e. end up not
        shared) if the shared parameters are not (auto-)wrapped under the same
        FSDP wrapper instance.

    Usage::

        with enable_wrap(**params):
            # Wraps children modules.
            self.l1 = auto_wrap(TransformerBlock())

    Args:
        module (nn.Module):
            module to wrap (if in :func:`enable_wrap` context)
        auto_wrap_policy (Callable):
            a function to determine should Module to be wrapped.
            (default: wrap if > 100M parameters)
    """
    if ConfigAutoWrap.in_autowrap_context:
        wrapped_module, remainder = ConfigAutoWrap.recursive_wrap(module, auto_wrap_policy=auto_wrap_policy, module_is_root=True, **kwargs)
        return wrapped_module
    return module