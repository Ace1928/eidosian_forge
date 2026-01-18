import dataclasses
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
def apply_to_collection(data: Any, dtype: Union[type, Any, Tuple[Union[type, Any]]], function: Callable, *args: Any, wrong_dtype: Optional[Union[type, Tuple[type, ...]]]=None, include_none: bool=True, allow_frozen: bool=False, **kwargs: Any) -> Any:
    """Recursively applies a function to all elements of a certain dtype.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections
            is of the ``wrong_dtype`` even if it is of type ``dtype``
        include_none: Whether to include an element if the output of ``function`` is ``None``.
        allow_frozen: Whether not to error upon encountering a frozen dataclass instance.
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        The resulting collection

    """
    if include_none is False or wrong_dtype is not None or allow_frozen is True:
        return _apply_to_collection_slow(data, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, allow_frozen=allow_frozen, **kwargs)
    if isinstance(data, dtype):
        return function(data, *args, **kwargs)
    if data.__class__ is list and all((isinstance(x, dtype) for x in data)):
        return [function(x, *args, **kwargs) for x in data]
    if data.__class__ is tuple and all((isinstance(x, dtype) for x in data)):
        return tuple((function(x, *args, **kwargs) for x in data))
    if data.__class__ is dict and all((isinstance(x, dtype) for x in data.values())):
        return {k: function(v, *args, **kwargs) for k, v in data.items()}
    return _apply_to_collection_slow(data, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, allow_frozen=allow_frozen, **kwargs)