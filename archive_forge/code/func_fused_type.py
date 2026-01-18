from __future__ import absolute_import
import math, sys
def fused_type(*args):
    if not args:
        raise TypeError('Expected at least one type as argument')
    rank = -1
    for type in args:
        if type not in (py_int, py_long, py_float, py_complex):
            break
        if type_ordering.index(type) > rank:
            result_type = type
    else:
        return result_type
    return _FusedType()