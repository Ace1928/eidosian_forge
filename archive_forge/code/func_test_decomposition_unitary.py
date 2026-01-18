from typing import List
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_decomposition_unitary(n):
    diagonal_angles = np.random.randn(2 ** n)
    diagonal_gate = cirq.DiagonalGate(diagonal_angles)
    decomposed_circ = cirq.Circuit(cirq.decompose(diagonal_gate(*cirq.LineQubit.range(n))))
    expected_f = [np.exp(1j * angle) for angle in diagonal_angles]
    decomposed_f = cirq.unitary(decomposed_circ).diagonal()
    np.testing.assert_allclose(decomposed_f, expected_f)