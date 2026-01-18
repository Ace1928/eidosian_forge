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
def test_array_printer():
    A = ArraySymbol('A', (4, 4, 6, 6, 6))
    I = IndexedBase('I')
    i, j, k = (Idx('i', (0, 1)), Idx('j', (2, 3)), Idx('k', (4, 5)))
    prntr = NumPyPrinter()
    assert prntr.doprint(ZeroArray(5)) == 'numpy.zeros((5,))'
    assert prntr.doprint(OneArray(5)) == 'numpy.ones((5,))'
    assert prntr.doprint(ArrayContraction(A, [2, 3])) == 'numpy.einsum("abccd->abd", A)'
    assert prntr.doprint(I) == 'I'
    assert prntr.doprint(ArrayDiagonal(A, [2, 3, 4])) == 'numpy.einsum("abccc->abc", A)'
    assert prntr.doprint(ArrayDiagonal(A, [0, 1], [2, 3])) == 'numpy.einsum("aabbc->cab", A)'
    assert prntr.doprint(ArrayContraction(A, [2], [3])) == 'numpy.einsum("abcde->abe", A)'
    assert prntr.doprint(Assignment(I[i, j, k], I[i, j, k])) == 'I = I'
    prntr = TensorflowPrinter()
    assert prntr.doprint(ZeroArray(5)) == 'tensorflow.zeros((5,))'
    assert prntr.doprint(OneArray(5)) == 'tensorflow.ones((5,))'
    assert prntr.doprint(ArrayContraction(A, [2, 3])) == 'tensorflow.linalg.einsum("abccd->abd", A)'
    assert prntr.doprint(I) == 'I'
    assert prntr.doprint(ArrayDiagonal(A, [2, 3, 4])) == 'tensorflow.linalg.einsum("abccc->abc", A)'
    assert prntr.doprint(ArrayDiagonal(A, [0, 1], [2, 3])) == 'tensorflow.linalg.einsum("aabbc->cab", A)'
    assert prntr.doprint(ArrayContraction(A, [2], [3])) == 'tensorflow.linalg.einsum("abcde->abe", A)'
    assert prntr.doprint(Assignment(I[i, j, k], I[i, j, k])) == 'I = I'