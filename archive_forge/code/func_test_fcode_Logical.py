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
def test_fcode_Logical():
    x, y, z = symbols('x y z')
    assert fcode(Not(x), source_format='free') == '.not. x'
    assert fcode(And(x, y), source_format='free') == 'x .and. y'
    assert fcode(And(x, Not(y)), source_format='free') == 'x .and. .not. y'
    assert fcode(And(Not(x), y), source_format='free') == 'y .and. .not. x'
    assert fcode(And(Not(x), Not(y)), source_format='free') == '.not. x .and. .not. y'
    assert fcode(Not(And(x, y), evaluate=False), source_format='free') == '.not. (x .and. y)'
    assert fcode(Or(x, y), source_format='free') == 'x .or. y'
    assert fcode(Or(x, Not(y)), source_format='free') == 'x .or. .not. y'
    assert fcode(Or(Not(x), y), source_format='free') == 'y .or. .not. x'
    assert fcode(Or(Not(x), Not(y)), source_format='free') == '.not. x .or. .not. y'
    assert fcode(Not(Or(x, y), evaluate=False), source_format='free') == '.not. (x .or. y)'
    assert fcode(And(Or(y, z), x), source_format='free') == 'x .and. (y .or. z)'
    assert fcode(And(Or(z, x), y), source_format='free') == 'y .and. (x .or. z)'
    assert fcode(And(Or(x, y), z), source_format='free') == 'z .and. (x .or. y)'
    assert fcode(Or(And(y, z), x), source_format='free') == 'x .or. y .and. z'
    assert fcode(Or(And(z, x), y), source_format='free') == 'y .or. x .and. z'
    assert fcode(Or(And(x, y), z), source_format='free') == 'z .or. x .and. y'
    assert fcode(And(x, y, z), source_format='free') == 'x .and. y .and. z'
    assert fcode(And(x, y, Not(z)), source_format='free') == 'x .and. y .and. .not. z'
    assert fcode(And(x, Not(y), z), source_format='free') == 'x .and. z .and. .not. y'
    assert fcode(And(Not(x), y, z), source_format='free') == 'y .and. z .and. .not. x'
    assert fcode(Not(And(x, y, z), evaluate=False), source_format='free') == '.not. (x .and. y .and. z)'
    assert fcode(Or(x, y, z), source_format='free') == 'x .or. y .or. z'
    assert fcode(Or(x, y, Not(z)), source_format='free') == 'x .or. y .or. .not. z'
    assert fcode(Or(x, Not(y), z), source_format='free') == 'x .or. z .or. .not. y'
    assert fcode(Or(Not(x), y, z), source_format='free') == 'y .or. z .or. .not. x'
    assert fcode(Not(Or(x, y, z), evaluate=False), source_format='free') == '.not. (x .or. y .or. z)'