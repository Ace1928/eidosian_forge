from sympy.assumptions.ask import (Q, ask)
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.simplify import simplify
from sympy.core.symbol import symbols
from sympy.matrices.expressions.fourier import DFT, IDFT
from sympy.matrices import det, Matrix, Identity
from sympy.testing.pytest import raises
def test_dft_creation():
    assert DFT(2)
    assert DFT(0)
    raises(ValueError, lambda: DFT(-1))
    raises(ValueError, lambda: DFT(2.0))
    raises(ValueError, lambda: DFT(2 + 1j))
    n = symbols('n')
    assert DFT(n)
    n = symbols('n', integer=False)
    raises(ValueError, lambda: DFT(n))
    n = symbols('n', negative=True)
    raises(ValueError, lambda: DFT(n))