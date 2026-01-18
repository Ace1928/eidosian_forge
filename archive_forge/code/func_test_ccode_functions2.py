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
def test_ccode_functions2():
    assert ccode(ceiling(x)) == 'ceil(x)'
    assert ccode(Abs(x)) == 'fabs(x)'
    assert ccode(gamma(x)) == 'tgamma(x)'
    r, s = symbols('r,s', real=True)
    assert ccode(Mod(ceiling(r), ceiling(s))) == '((ceil(r) % ceil(s)) + ceil(s)) % ceil(s)'
    assert ccode(Mod(r, s)) == 'fmod(r, s)'
    p1, p2 = symbols('p1 p2', integer=True, positive=True)
    assert ccode(Mod(p1, p2)) == 'p1 % p2'
    assert ccode(Mod(p1, p2 + 3)) == 'p1 % (p2 + 3)'
    assert ccode(Mod(-3, -7, evaluate=False)) == '(-3) % (-7)'
    assert ccode(-Mod(3, 7, evaluate=False)) == '-(3 % 7)'
    assert ccode(r * Mod(p1, p2)) == 'r*(p1 % p2)'
    assert ccode(Mod(p1, p2) ** s) == 'pow(p1 % p2, s)'
    n = symbols('n', integer=True, negative=True)
    assert ccode(Mod(-n, p2)) == '(-n) % p2'
    assert ccode(fibonacci(n)) == '(1.0/5.0)*pow(2, -n)*sqrt(5)*(-pow(1 - sqrt(5), n) + pow(1 + sqrt(5), n))'
    assert ccode(lucas(n)) == 'pow(2, -n)*(pow(1 - sqrt(5), n) + pow(1 + sqrt(5), n))'