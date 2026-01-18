from typing import List
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_decomposition_diagonal_exponent(n):
    diagonal_angles = np.random.randn(2 ** n)
    diagonal_gate = cirq.DiagonalGate(diagonal_angles)
    sqrt_diagonal_gate = diagonal_gate ** 0.5
    decomposed_circ = cirq.Circuit(cirq.decompose(sqrt_diagonal_gate(*cirq.LineQubit.range(n))))
    expected_f = [np.exp(1j * angle / 2) for angle in diagonal_angles]
    decomposed_f = cirq.unitary(decomposed_circ).diagonal()
    np.testing.assert_allclose(decomposed_f, expected_f)