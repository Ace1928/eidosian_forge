import contextlib
from typing import Any, Callable, Dict, Generator, Optional, Set, Tuple, Type, cast
import torch.nn as nn
def default_auto_wrap_policy(module: nn.Module, recurse: bool, unwrapped_params: int, module_is_root: bool, min_num_params: int=int(100000000.0), force_leaf_modules: Optional[Set[Type[nn.Module]]]=None, exclude_wrap_modules: Optional[Set[Type[nn.Module]]]=None, skip_params_check_for_root: bool=False) -> bool:
    """Default policy function for :func:`auto_wrap`.

       Return if a module should be wrapped during :func:`auto_wrap`.

       The first four parameters are used by :func:`auto_wrap`. If
       you write a custom version of this policy function, your version
       needs to at least accept the first four parameters and free
       to do whatever you want in the function.

    Args:
       module (nn.Module):
           The module to be considered in this decision.
       recurse (bool):
           Indicate if this is called to make a decision on whether we
           should recurse down a subgraph of the module structure.
           If False, it means this function is called to make a decision
           on whether we should wrap the said module.
       unwrapped_params (int):
           The number of parameters yet to be wrapped in this module.
       module_is_root (bool):
           Indicates if current module is the root.

       min_num_params (int):
           Customizable policy input. It controls the size threshold
           on how big should a module be to be considered wrapped.
       force_leaf_modules (Set[Type[nn.Module]]): set of module types to
           keep as leaves, i.e., their children will never be wrapped.
       exclude_wrap_modules (Set[Type[nn.Module]]):
           Customizable set of module types to be excluded in wrapping.
       skip_params_check_for_root (bool):
           If module_is_root is True, then this includes the root in
           wrapping regardless of their number of unwrapped params.
    """
    force_leaf_modules = default_auto_wrap_policy.FORCE_LEAF_MODULES if force_leaf_modules is None else force_leaf_modules
    exclude_wrap_modules = default_auto_wrap_policy.EXCLUDE_WRAP_MODULES if exclude_wrap_modules is None else exclude_wrap_modules
    is_large = unwrapped_params >= min_num_params
    if recurse:
        return is_large and (not isinstance(module, tuple(force_leaf_modules)))
    else:
        return (module_is_root and skip_params_check_for_root or is_large) and (not isinstance(module, tuple(exclude_wrap_modules)))