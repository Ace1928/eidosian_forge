from sympy.codegen import Assignment
from sympy.codegen.ast import none
from sympy.codegen.cfunctions import expm1, log1p
from sympy.codegen.scipy_nodes import cosm1
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core import Expr, Mod, symbols, Eq, Le, Gt, zoo, oo, Rational, Pow
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions import acos, KroneckerDelta, Piecewise, sign, sqrt, Min, Max, cot, acsch, asec, coth
from sympy.logic import And, Or
from sympy.matrices import SparseMatrix, MatrixSymbol, Identity
from sympy.printing.pycode import (
from sympy.printing.tensorflow import TensorflowPrinter
from sympy.printing.numpy import NumPyPrinter, SciPyPrinter
from sympy.testing.pytest import raises, skip
from sympy.tensor import IndexedBase, Idx
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayDiagonal, ArrayContraction, ZeroArray, OneArray
from sympy.external import import_module
from sympy.functions.special.gamma_functions import loggamma
def test_issue_18770():
    numpy = import_module('numpy')
    if not numpy:
        skip('numpy not installed.')
    from sympy.functions.elementary.miscellaneous import Max, Min
    from sympy.utilities.lambdify import lambdify
    expr1 = Min(0.1 * x + 3, x + 1, 0.5 * x + 1)
    func = lambdify(x, expr1, 'numpy')
    assert (func(numpy.linspace(0, 3, 3)) == [1.0, 1.75, 2.5]).all()
    assert func(4) == 3
    expr1 = Max(x ** 2, x ** 3)
    func = lambdify(x, expr1, 'numpy')
    assert (func(numpy.linspace(-1, 2, 4)) == [1, 0, 1, 8]).all()
    assert func(4) == 64