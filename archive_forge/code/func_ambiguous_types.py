from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def ambiguous_types(ty0, ty1):
    from numpy import complex64, complex128
    from numpy import float32, float64
    from numpy import int8, int16, int32, int64, intp, intc
    from numpy import uint8, uint16, uint32, uint64, uintp, uintc
    try:
        from numpy import complex256, float128
    except ImportError:
        complex256 = complex128
        float128 = float64
    if isinstance(ty0, tuple):
        if len(ty0) != len(ty1):
            return False
        return all((ambiguous_types(t0, t1) for t0, t1 in zip(ty0, ty1)))
    ambiguous_float_types = (float, float64)
    if ty0 in ambiguous_float_types and ty1 in ambiguous_float_types:
        return True
    ambiguous_cplx_types = (complex, complex128)
    if ty0 in ambiguous_cplx_types and ty1 in ambiguous_cplx_types:
        return True
    ambiguous_int_types = (int64, int)
    if ty0 in ambiguous_int_types and ty1 in ambiguous_int_types:
        return True
    if type(ty0) is not type(ty1):
        return False
    if not hasattr(ty0, '__args__'):
        return False
    if type(ty0) is NDArray:
        return ambiguous_types(ty0.__args__[1:], ty1.__args__[1:])
    else:
        return ambiguous_types(ty0.__args__, ty1.__args__)