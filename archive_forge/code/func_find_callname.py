import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def find_callname(func_ir, expr, typemap=None, definition_finder=get_definition):
    """Try to find a call expression's function and module names and return
    them as strings for unbounded calls. If the call is a bounded call, return
    the self object instead of module name. Raise GuardException if failed.

    Providing typemap can make the call matching more accurate in corner cases
    such as bounded call on an object which is inside another object.
    """
    require(isinstance(expr, ir.Expr) and expr.op == 'call')
    callee = expr.func
    callee_def = definition_finder(func_ir, callee)
    attrs = []
    obj = None
    while True:
        if isinstance(callee_def, (ir.Global, ir.FreeVar)):
            keys = ['name', '_name', '__name__']
            value = None
            for key in keys:
                if hasattr(callee_def.value, key):
                    value = getattr(callee_def.value, key)
                    break
            if not value or not isinstance(value, str):
                raise GuardException
            attrs.append(value)
            def_val = callee_def.value
            if isinstance(def_val, _Intrinsic):
                def_val = def_val._defn
            if hasattr(def_val, '__module__'):
                mod_name = def_val.__module__
                mod_not_none = mod_name is not None
                numpy_toplevel = mod_not_none and (mod_name == 'numpy' or mod_name.startswith('numpy.'))
                if numpy_toplevel and hasattr(numpy, value) and (def_val == getattr(numpy, value)):
                    attrs += ['numpy']
                elif hasattr(numpy.random, value) and def_val == getattr(numpy.random, value):
                    attrs += ['random', 'numpy']
                elif mod_not_none:
                    attrs.append(mod_name)
            else:
                class_name = def_val.__class__.__name__
                if class_name == 'builtin_function_or_method':
                    class_name = 'builtin'
                if class_name != 'module':
                    attrs.append(class_name)
            break
        elif isinstance(callee_def, ir.Expr) and callee_def.op == 'getattr':
            obj = callee_def.value
            attrs.append(callee_def.attr)
            if typemap and obj.name in typemap:
                typ = typemap[obj.name]
                if not isinstance(typ, types.Module):
                    return (attrs[0], obj)
            callee_def = definition_finder(func_ir, obj)
        else:
            if obj is not None:
                return ('.'.join(reversed(attrs)), obj)
            raise GuardException
    return (attrs[0], '.'.join(reversed(attrs[1:])))