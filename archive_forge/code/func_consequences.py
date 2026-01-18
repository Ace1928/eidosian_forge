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
def consequences(self, assumptions, variables):
    """Determine fixed values for the variables based on the solver state and assumptions.
        >>> s = Solver()
        >>> a, b, c, d = Bools('a b c d')
        >>> s.add(Implies(a,b), Implies(b, c))
        >>> s.consequences([a],[b,c,d])
        (sat, [Implies(a, b), Implies(a, c)])
        >>> s.consequences([Not(c),d],[a,b,c,d])
        (sat, [Implies(d, d), Implies(Not(c), Not(c)), Implies(Not(c), Not(b)), Implies(Not(c), Not(a))])
        """
    if isinstance(assumptions, list):
        _asms = AstVector(None, self.ctx)
        for a in assumptions:
            _asms.push(a)
        assumptions = _asms
    if isinstance(variables, list):
        _vars = AstVector(None, self.ctx)
        for a in variables:
            _vars.push(a)
        variables = _vars
    _z3_assert(isinstance(assumptions, AstVector), 'ast vector expected')
    _z3_assert(isinstance(variables, AstVector), 'ast vector expected')
    consequences = AstVector(None, self.ctx)
    r = Z3_solver_get_consequences(self.ctx.ref(), self.solver, assumptions.vector, variables.vector, consequences.vector)
    sz = len(consequences)
    consequences = [consequences[i] for i in range(sz)]
    return (CheckSatResult(r), consequences)