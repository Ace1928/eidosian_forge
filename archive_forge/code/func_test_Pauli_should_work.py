from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.physics.paulialgebra import Pauli
from sympy.testing.pytest import XFAIL
from sympy.physics.quantum import TensorProduct
@XFAIL
def test_Pauli_should_work():
    assert sigma1 * sigma3 * sigma1 == -sigma3