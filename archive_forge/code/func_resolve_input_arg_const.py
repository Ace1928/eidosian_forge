import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def resolve_input_arg_const(input_arg_idx):
    """
        Resolves an input arg to a constant (if possible)
        """
    input_arg_ty = called_args[input_arg_idx]
    if isinstance(input_arg_ty, types.NoneType):
        return input_arg_ty
    if isinstance(input_arg_ty, types.Omitted):
        val = input_arg_ty.value
        if isinstance(val, types.NoneType):
            return val
        elif val is None:
            return types.NoneType('none')
    return getattr(input_arg_ty, 'literal_type', Unknown())