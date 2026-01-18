from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda, diff)
from sympy.core.mod import Mod
from sympy.core import (Catalan, EulerGamma, GoldenRatio)
from sympy.core.numbers import (E, Float, I, Integer, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (conjugate, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.sets.fancysets import Range
from sympy.codegen import For, Assignment, aug_assign
from sympy.codegen.ast import Declaration, Variable, float32, float64, \
from sympy.core.expr import UnevaluatedExpr
from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Or, Not, Equivalent, Xor
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.fortran import fcode, FCodePrinter
from sympy.tensor import IndexedBase, Idx
from sympy.tensor.array.expressions import ArraySymbol, ArrayElement
from sympy.utilities.lambdify import implemented_function
from sympy.testing.pytest import raises
def test_derived_classes():

    class MyFancyFCodePrinter(FCodePrinter):
        _default_settings = FCodePrinter._default_settings.copy()
    printer = MyFancyFCodePrinter()
    x = symbols('x')
    assert printer.doprint(sin(x), 'bork') == '      bork = sin(x)'