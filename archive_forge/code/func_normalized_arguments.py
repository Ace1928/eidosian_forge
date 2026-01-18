from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict, Set
from ._compatibility import compatibility
from .immutable_collections import immutable_dict, immutable_list
import torch
import builtins
import types
import inspect
import warnings
from torch.fx.operator_schemas import normalize_function, normalize_module, ArgsKwargsPair
from .._ops import ops as _ops
@compatibility(is_backward_compatible=False)
def normalized_arguments(self, root: torch.nn.Module, arg_types: Optional[Tuple[Any]]=None, kwarg_types: Optional[Dict[str, Any]]=None, normalize_to_only_use_kwargs: bool=False) -> Optional[ArgsKwargsPair]:
    """
        Returns normalized arguments to Python targets. This means that
        `args/kwargs` will be matched up to the module/functional's
        signature and return exclusively kwargs in positional order
        if `normalize_to_only_use_kwargs` is true.
        Also populates default values. Does not support positional-only
        parameters or varargs parameters.

        Supports module calls.

        May require `arg_types` and `kwarg_types` in order to disambiguate overloads.

        Args:
            root (torch.nn.Module): Module upon which to resolve module targets.
            arg_types (Optional[Tuple[Any]]): Tuple of arg types for the args
            kwarg_types (Optional[Dict[str, Any]]): Dict of arg types for the kwargs
            normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

        Returns:

            Returns NamedTuple ArgsKwargsPair, or `None` if not successful.
        """
    if self.op == 'call_function':
        assert callable(self.target)
        return normalize_function(self.target, self.args, self.kwargs, arg_types, kwarg_types)
    elif self.op == 'call_module':
        assert isinstance(self.target, str)
        return normalize_module(root, self.target, self.args, self.kwargs)
    return None