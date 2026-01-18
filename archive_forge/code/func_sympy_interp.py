import functools
from typing import Any, Dict, Union
import sympy
from sympy.logic.boolalg import Boolean as SympyBoolean, BooleanAtom
import torch
from .functions import (
def sympy_interp(analysis, env: Dict[sympy.Symbol, Any], expr: Union[sympy.Expr, SympyBoolean]):
    dtype = None
    if isinstance(expr, BooleanAtom):
        dtype = torch.bool
    elif isinstance(expr, sympy.Integer):
        dtype = torch.int64
    elif isinstance(expr, sympy.Number):
        dtype = torch.double
    if dtype is not None:
        return analysis.constant(expr, dtype)
    elif isinstance(expr, sympy.Symbol):
        return env[expr]
    if isinstance(expr, sympy.Pow) and isinstance(expr.args[1], sympy.core.numbers.Half):
        return analysis.sqrt(sympy_interp(analysis, env, expr.args[0]))
    args = [sympy_interp(analysis, env, arg) for arg in expr.args]
    handler_name = handlers()[expr.func]
    handler = getattr(analysis, handler_name)
    if handler_name in ASSOCIATIVE_OPS:
        assert len(args) > 1
        acc = handler(args[0], args[1])
        for i in range(2, len(args)):
            acc = handler(acc, args[i])
        return acc
    else:
        return handler(*args)