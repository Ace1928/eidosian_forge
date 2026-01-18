from itertools import chain
from .coretypes import (Unit, int8, int16, int32, int64, uint8, uint16, uint32,
def matches_typeset(types, signature):
    """Match argument types to the parameter types of a signature

    >>> matches_typeset(int32, integral)
    True
    >>> matches_typeset(float32, integral)
    False
    >>> matches_typeset(integral, real)
    True
    """
    if types in signature:
        return True
    match = True
    for a, b in zip(types, signature):
        check = isinstance(b, TypeSet)
        if check and a not in b or (not check and a != b):
            match = False
            break
    return match