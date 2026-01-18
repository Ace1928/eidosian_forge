import typing
import sympy
from sympy.core import Add, Mul
from sympy.core import Symbol, Expr, Float, Rational, Integer, Basic
from sympy.core.function import UndefinedFunction, Function
from sympy.core.relational import Relational, Unequality, Equality, LessThan, GreaterThan, StrictLessThan, StrictGreaterThan
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log, Pow
from sympy.functions.elementary.hyperbolic import sinh, cosh, tanh
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin, cos, tan, asin, acos, atan, atan2
from sympy.logic.boolalg import And, Or, Xor, Implies, Boolean
from sympy.logic.boolalg import BooleanTrue, BooleanFalse, BooleanFunction, Not, ITE
from sympy.printing.printer import Printer
from sympy.sets import Interval
def safe_update(syms: set, inf):
    for s in syms:
        assert s.is_Symbol
        if (old_type := _symbols.setdefault(s, inf)) != inf:
            raise TypeError(f'Could not infer type of `{s}`. Apparently both `{old_type}` and `{inf}`?')