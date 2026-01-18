import sys
import types
import typing
import typing_extensions
from mypy_extensions import _TypedDictMeta as _TypedDictMeta_Mypy
def get_last_origin(tp):
    """Get the last base of (multiply) subscripted type. Supports generic types,
    Union, Callable, and Tuple. Returns None for unsupported types.
    Examples::

        get_last_origin(int) == None
        get_last_origin(ClassVar[int]) == None
        get_last_origin(Generic[T]) == Generic
        get_last_origin(Union[T, int][str]) == Union[T, int]
        get_last_origin(List[Tuple[T, T]][int]) == List[Tuple[T, T]]
        get_last_origin(List) == List
    """
    if NEW_TYPING:
        raise ValueError('This function is only supported in Python 3.6, use get_origin instead')
    sentinel = object()
    origin = getattr(tp, '__origin__', sentinel)
    if origin is sentinel:
        return None
    if origin is None:
        return tp
    return origin