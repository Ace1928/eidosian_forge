import contextlib
from typing import Any, Callable, Dict, Generator, Optional, Set, Tuple, Type, cast
import torch.nn as nn
@staticmethod
def recursive_wrap(module: nn.Module, auto_wrap_policy: Optional[Callable], module_is_root: bool, **kwargs: Any) -> Tuple[nn.Module, int]:
    """
        Automatically wrap child modules of *module* that meet the given
        criteria with :func:`auto_wrap`.

        Args:
            module (nn.Module):
                module to recursively wrap
            auto_wrap_policy (Callable, Optional):
                optionally, override the :func:`auto_wrap_policy` from the context.

        Returns:
            (nn.Module, int):
                Wrapped module and the number parameters wrapped recursively.
        """
    if auto_wrap_policy is None:
        auto_wrap_policy = ConfigAutoWrap.auto_wrap_policy
    for _, child in module.named_modules():
        assert not isinstance(child, cast(type, ConfigAutoWrap.wrapper_cls))
    num_params = sum([p.numel() for p in module.parameters()])
    assert auto_wrap_policy is not None
    if auto_wrap_policy(module=module, recurse=True, unwrapped_params=num_params, module_is_root=module_is_root):
        total_wrapped_params = 0
        for name, child in module.named_children():
            wrapped_child, num_wrapped_params = ConfigAutoWrap.recursive_wrap(module=child, auto_wrap_policy=auto_wrap_policy, module_is_root=False, **kwargs)
            setattr(module, name, wrapped_child)
            total_wrapped_params += num_wrapped_params
        remainder = num_params - total_wrapped_params
        if auto_wrap_policy(module=module, recurse=False, unwrapped_params=remainder, module_is_root=module_is_root):
            return (wrap(module, **kwargs), num_params)
        else:
            return (module, total_wrapped_params)
    return (module, 0)