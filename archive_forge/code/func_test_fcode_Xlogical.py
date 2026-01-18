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
def test_fcode_Xlogical():
    x, y, z = symbols('x y z')
    assert fcode(Xor(x, y, evaluate=False), source_format='free') == 'x .neqv. y'
    assert fcode(Xor(x, Not(y), evaluate=False), source_format='free') == 'x .neqv. .not. y'
    assert fcode(Xor(Not(x), y, evaluate=False), source_format='free') == 'y .neqv. .not. x'
    assert fcode(Xor(Not(x), Not(y), evaluate=False), source_format='free') == '.not. x .neqv. .not. y'
    assert fcode(Not(Xor(x, y, evaluate=False), evaluate=False), source_format='free') == '.not. (x .neqv. y)'
    assert fcode(Equivalent(x, y), source_format='free') == 'x .eqv. y'
    assert fcode(Equivalent(x, Not(y)), source_format='free') == 'x .eqv. .not. y'
    assert fcode(Equivalent(Not(x), y), source_format='free') == 'y .eqv. .not. x'
    assert fcode(Equivalent(Not(x), Not(y)), source_format='free') == '.not. x .eqv. .not. y'
    assert fcode(Not(Equivalent(x, y), evaluate=False), source_format='free') == '.not. (x .eqv. y)'
    assert fcode(Equivalent(And(y, z), x), source_format='free') == 'x .eqv. y .and. z'
    assert fcode(Equivalent(And(z, x), y), source_format='free') == 'y .eqv. x .and. z'
    assert fcode(Equivalent(And(x, y), z), source_format='free') == 'z .eqv. x .and. y'
    assert fcode(And(Equivalent(y, z), x), source_format='free') == 'x .and. (y .eqv. z)'
    assert fcode(And(Equivalent(z, x), y), source_format='free') == 'y .and. (x .eqv. z)'
    assert fcode(And(Equivalent(x, y), z), source_format='free') == 'z .and. (x .eqv. y)'
    assert fcode(Equivalent(Or(y, z), x), source_format='free') == 'x .eqv. y .or. z'
    assert fcode(Equivalent(Or(z, x), y), source_format='free') == 'y .eqv. x .or. z'
    assert fcode(Equivalent(Or(x, y), z), source_format='free') == 'z .eqv. x .or. y'
    assert fcode(Or(Equivalent(y, z), x), source_format='free') == 'x .or. (y .eqv. z)'
    assert fcode(Or(Equivalent(z, x), y), source_format='free') == 'y .or. (x .eqv. z)'
    assert fcode(Or(Equivalent(x, y), z), source_format='free') == 'z .or. (x .eqv. y)'
    assert fcode(Equivalent(Xor(y, z, evaluate=False), x), source_format='free') == 'x .eqv. (y .neqv. z)'
    assert fcode(Equivalent(Xor(z, x, evaluate=False), y), source_format='free') == 'y .eqv. (x .neqv. z)'
    assert fcode(Equivalent(Xor(x, y, evaluate=False), z), source_format='free') == 'z .eqv. (x .neqv. y)'
    assert fcode(Xor(Equivalent(y, z), x, evaluate=False), source_format='free') == 'x .neqv. (y .eqv. z)'
    assert fcode(Xor(Equivalent(z, x), y, evaluate=False), source_format='free') == 'y .neqv. (x .eqv. z)'
    assert fcode(Xor(Equivalent(x, y), z, evaluate=False), source_format='free') == 'z .neqv. (x .eqv. y)'
    assert fcode(Xor(And(y, z), x, evaluate=False), source_format='free') == 'x .neqv. y .and. z'
    assert fcode(Xor(And(z, x), y, evaluate=False), source_format='free') == 'y .neqv. x .and. z'
    assert fcode(Xor(And(x, y), z, evaluate=False), source_format='free') == 'z .neqv. x .and. y'
    assert fcode(And(Xor(y, z, evaluate=False), x), source_format='free') == 'x .and. (y .neqv. z)'
    assert fcode(And(Xor(z, x, evaluate=False), y), source_format='free') == 'y .and. (x .neqv. z)'
    assert fcode(And(Xor(x, y, evaluate=False), z), source_format='free') == 'z .and. (x .neqv. y)'
    assert fcode(Xor(Or(y, z), x, evaluate=False), source_format='free') == 'x .neqv. y .or. z'
    assert fcode(Xor(Or(z, x), y, evaluate=False), source_format='free') == 'y .neqv. x .or. z'
    assert fcode(Xor(Or(x, y), z, evaluate=False), source_format='free') == 'z .neqv. x .or. y'
    assert fcode(Or(Xor(y, z, evaluate=False), x), source_format='free') == 'x .or. (y .neqv. z)'
    assert fcode(Or(Xor(z, x, evaluate=False), y), source_format='free') == 'y .or. (x .neqv. z)'
    assert fcode(Or(Xor(x, y, evaluate=False), z), source_format='free') == 'z .or. (x .neqv. y)'
    assert fcode(Xor(x, y, z, evaluate=False), source_format='free') == 'x .neqv. y .neqv. z'
    assert fcode(Xor(x, y, Not(z), evaluate=False), source_format='free') == 'x .neqv. y .neqv. .not. z'
    assert fcode(Xor(x, Not(y), z, evaluate=False), source_format='free') == 'x .neqv. z .neqv. .not. y'
    assert fcode(Xor(Not(x), y, z, evaluate=False), source_format='free') == 'y .neqv. z .neqv. .not. x'