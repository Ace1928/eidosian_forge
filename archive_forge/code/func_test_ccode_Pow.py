from sympy.core import (
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.functions import (
from sympy.sets import Range
from sympy.logic import ITE, Implies, Equivalent
from sympy.codegen import For, aug_assign, Assignment
from sympy.testing.pytest import raises, XFAIL
from sympy.printing.c import C89CodePrinter, C99CodePrinter, get_math_macros
from sympy.codegen.ast import (
from sympy.codegen.cfunctions import expm1, log1p, exp2, log2, fma, log10, Cbrt, hypot, Sqrt
from sympy.codegen.cnodes import restrict
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol, SparseMatrix
from sympy.printing.codeprinter import ccode
def test_ccode_Pow():
    assert ccode(x ** 3) == 'pow(x, 3)'
    assert ccode(x ** y ** 3) == 'pow(x, pow(y, 3))'
    g = implemented_function('g', Lambda(x, 2 * x))
    assert ccode(1 / (g(x) * 3.5) ** (x - y ** x) / (x ** 2 + y)) == 'pow(3.5*2*x, -x + pow(y, x))/(pow(x, 2) + y)'
    assert ccode(x ** (-1.0)) == '1.0/x'
    assert ccode(x ** Rational(2, 3)) == 'pow(x, 2.0/3.0)'
    assert ccode(x ** Rational(2, 3), type_aliases={real: float80}) == 'powl(x, 2.0L/3.0L)'
    _cond_cfunc = [(lambda base, exp: exp.is_integer, 'dpowi'), (lambda base, exp: not exp.is_integer, 'pow')]
    assert ccode(x ** 3, user_functions={'Pow': _cond_cfunc}) == 'dpowi(x, 3)'
    assert ccode(x ** 0.5, user_functions={'Pow': _cond_cfunc}) == 'pow(x, 0.5)'
    assert ccode(x ** Rational(16, 5), user_functions={'Pow': _cond_cfunc}) == 'pow(x, 16.0/5.0)'
    _cond_cfunc2 = [(lambda base, exp: base == 2, lambda base, exp: 'exp2(%s)' % exp), (lambda base, exp: base != 2, 'pow')]
    assert ccode(2 ** x, user_functions={'Pow': _cond_cfunc2}) == 'exp2(x)'
    assert ccode(x ** 2, user_functions={'Pow': _cond_cfunc2}) == 'pow(x, 2)'
    assert ccode(Mul(-2, x, Pow(Mul(y, y, evaluate=False), -1, evaluate=False), evaluate=False)) == '-2*x/(y*y)'