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
def decls(self):
    """Return a list with all symbols that have an interpretation in the model `self`.
        >>> f = Function('f', IntSort(), IntSort())
        >>> x = Int('x')
        >>> s = Solver()
        >>> s.add(x > 0, x < 2, f(x) == 0)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> m.decls()
        [x, f]
        """
    r = []
    for i in range(Z3_model_get_num_consts(self.ctx.ref(), self.model)):
        r.append(FuncDeclRef(Z3_model_get_const_decl(self.ctx.ref(), self.model, i), self.ctx))
    for i in range(Z3_model_get_num_funcs(self.ctx.ref(), self.model)):
        r.append(FuncDeclRef(Z3_model_get_func_decl(self.ctx.ref(), self.model, i), self.ctx))
    return r