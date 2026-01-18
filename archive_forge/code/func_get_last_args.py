import sys
import types
import typing
import typing_extensions
from mypy_extensions import _TypedDictMeta as _TypedDictMeta_Mypy
def get_last_args(tp):
    """Get last arguments of (multiply) subscripted type.
       Parameters for Callable are flattened. Examples::

        get_last_args(int) == ()
        get_last_args(Union) == ()
        get_last_args(ClassVar[int]) == (int,)
        get_last_args(Union[T, int]) == (T, int)
        get_last_args(Iterable[Tuple[T, S]][int, T]) == (int, T)
        get_last_args(Callable[[T], int]) == (T, int)
        get_last_args(Callable[[], int]) == (int,)
    """
    if NEW_TYPING:
        raise ValueError('This function is only supported in Python 3.6, use get_args instead')
    elif is_classvar(tp):
        return (tp.__type__,) if tp.__type__ is not None else ()
    elif is_generic_type(tp):
        try:
            if tp.__args__ is not None and len(tp.__args__) > 0:
                return tp.__args__
        except AttributeError:
            pass
        return tp.__parameters__ if tp.__parameters__ is not None else ()
    elif is_union_type(tp):
        try:
            return tp.__args__ if tp.__args__ is not None else ()
        except AttributeError:
            return tp.__union_params__ if tp.__union_params__ is not None else ()
    elif is_callable_type(tp):
        return tp.__args__ if tp.__args__ is not None else ()
    elif is_tuple_type(tp):
        try:
            return tp.__args__ if tp.__args__ is not None else ()
        except AttributeError:
            return tp.__tuple_params__ if tp.__tuple_params__ is not None else ()
    else:
        return ()