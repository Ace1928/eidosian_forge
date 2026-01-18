import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def lower_binop(self, expr, op, inplace=False):
    lhs = self.loadvar(expr.lhs.name)
    rhs = self.loadvar(expr.rhs.name)
    assert not isinstance(op, str)
    if op in PYTHON_BINOPMAP:
        fname, inplace = PYTHON_BINOPMAP[op]
        fn = getattr(self.pyapi, fname)
        res = fn(lhs, rhs, inplace=inplace)
    else:
        fn = PYTHON_COMPAREOPMAP.get(expr.fn, expr.fn)
        if fn == 'in':
            lhs, rhs = (rhs, lhs)
        res = self.pyapi.object_richcompare(lhs, rhs, fn)
    self.check_error(res)
    return res