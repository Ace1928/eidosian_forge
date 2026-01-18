from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
def maybe_parse_user_type(t):
    """Try to coerce a user-supplied type directive into a list of types.

    This function should be used in all places where a user specifies a type,
    for consistency.

    The policy for what defines valid user input should be clear from the implementation.
    """
    is_type = isinstance(t, type)
    is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
    is_string = isinstance(t, str)
    is_iterable = isinstance(t, Iterable)
    if is_preserved:
        return [t]
    elif is_string:
        return [t]
    elif is_type and (not is_iterable):
        return [t]
    elif is_iterable:
        ts = t
        return tuple((e for t in ts for e in maybe_parse_user_type(t)))
    else:
        raise TypeError('Type specifications must be types or strings. Input: {}'.format(t))