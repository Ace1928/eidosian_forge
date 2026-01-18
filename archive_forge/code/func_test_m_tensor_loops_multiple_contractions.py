from io import StringIO
from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import OctaveCodeGen, codegen, make_routine
from sympy.testing.pytest import raises
from sympy.testing.pytest import XFAIL
import sympy
def test_m_tensor_loops_multiple_contractions():
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m, o, p = symbols('n m o p', integer=True)
    A = IndexedBase('A')
    B = IndexedBase('B')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)
    result, = codegen(('tensorthing', Eq(y[i], B[j, k, l] * A[i, j, k, l])), 'Octave', header=False, empty=False)
    source = result[1]
    expected = 'function y = tensorthing(A, B, m, n, o, p)\n  for i = 1:m\n    y(i) = 0;\n  end\n  for i = 1:m\n    for j = 1:n\n      for k = 1:o\n        for l = 1:p\n          y(i) = A(i, j, k, l).*B(j, k, l) + y(i);\n        end\n      end\n    end\n  end\nend\n'
    assert source == expected