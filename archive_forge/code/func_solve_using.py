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
def solve_using(s, *args, **keywords):
    """Solve the constraints `*args` using solver `s`.

    This is a simple function for creating demonstrations. It is similar to `solve`,
    but it uses the given solver `s`.
    It configures solver `s` using the options in `keywords`, adds the constraints
    in `args`, and invokes check.
    """
    show = keywords.pop('show', False)
    if z3_debug():
        _z3_assert(isinstance(s, Solver), 'Solver object expected')
    s.set(**keywords)
    s.add(*args)
    if show:
        print('Problem:')
        print(s)
    r = s.check()
    if r == unsat:
        print('no solution')
    elif r == unknown:
        print('failed to solve')
        try:
            print(s.model())
        except Z3Exception:
            return
    else:
        if show:
            print('Solution:')
        print(s.model())