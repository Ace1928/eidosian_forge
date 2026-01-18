import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def rewrite_array_ndim(val, func_ir, called_args):
    if getattr(val, 'op', None) == 'getattr':
        if val.attr == 'ndim':
            arg_def = guard(get_definition, func_ir, val.value)
            if isinstance(arg_def, ir.Arg):
                argty = called_args[arg_def.index]
                if isinstance(argty, types.Array):
                    rewrite_statement(func_ir, stmt, argty.ndim)