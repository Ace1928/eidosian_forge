from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.physics.paulialgebra import Pauli
from sympy.testing.pytest import XFAIL
from sympy.physics.quantum import TensorProduct
def test_evaluate_pauli_product():
    from sympy.physics.paulialgebra import evaluate_pauli_product
    assert evaluate_pauli_product(I * sigma2 * sigma3) == -sigma1
    assert evaluate_pauli_product(-I * 4 * sigma1 * sigma2) == 4 * sigma3
    assert evaluate_pauli_product(1 + I * sigma1 * sigma2 * sigma1 * sigma2 + I * sigma1 * sigma2 * tau1 * sigma1 * sigma3 + (tau1 ** 2).subs(tau1, I * sigma1) + sigma3 * (tau1 ** 2).subs(tau1, I * sigma1) + TensorProduct(I * sigma1 * sigma2 * sigma1 * sigma2, 1)) == 1 - I + I * sigma3 * tau1 * sigma2 - 1 - sigma3 - I * TensorProduct(1, 1)