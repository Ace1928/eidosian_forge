import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
@classmethod
def literal_array(cls, elems):
    """
        Construct a literal array constant made of the given members.
        """
    tys = [el.type for el in elems]
    if len(tys) == 0:
        raise ValueError('need at least one element')
    ty = tys[0]
    for other in tys:
        if ty != other:
            raise TypeError('all elements must have the same type')
    return cls(types.ArrayType(ty, len(elems)), elems)