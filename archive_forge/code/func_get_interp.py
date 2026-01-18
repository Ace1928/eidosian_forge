from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def get_interp(self, decl):
    """Return the interpretation for a given declaration or constant.

        >>> f = Function('f', IntSort(), IntSort())
        >>> x = Int('x')
        >>> s = Solver()
        >>> s.add(x > 0, x < 2, f(x) == 0)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m[x]
        1
        >>> m[f]
        [else -> 0]
        """
    if z3_debug():
        _z3_assert(isinstance(decl, FuncDeclRef) or is_const(decl), 'Z3 declaration expected')
    if is_const(decl):
        decl = decl.decl()
    try:
        if decl.arity() == 0:
            _r = Z3_model_get_const_interp(self.ctx.ref(), self.model, decl.ast)
            if _r.value is None:
                return None
            r = _to_expr_ref(_r, self.ctx)
            if is_as_array(r):
                fi = self.get_interp(get_as_array_func(r))
                if fi is None:
                    return fi
                e = fi.else_value()
                if e is None:
                    return fi
                if fi.arity() != 1:
                    return fi
                srt = decl.range()
                dom = srt.domain()
                e = K(dom, e)
                i = 0
                sz = fi.num_entries()
                n = fi.arity()
                while i < sz:
                    fe = fi.entry(i)
                    e = Store(e, fe.arg_value(0), fe.value())
                    i += 1
                return e
            else:
                return r
        else:
            return FuncInterp(Z3_model_get_func_interp(self.ctx.ref(), self.model, decl.ast), self.ctx)
    except Z3Exception:
        return None