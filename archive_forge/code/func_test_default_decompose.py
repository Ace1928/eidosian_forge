import itertools
import pytest
import numpy as np
import sympy
import cirq
@pytest.mark.parametrize('paulis,phase_exponent_negative,sign', itertools.product(itertools.product((cirq.X, cirq.Y, cirq.Z, None), repeat=3), (0, 0.1, 0.5, 1, -0.25), (+1, -1)))
def test_default_decompose(paulis, phase_exponent_negative: float, sign: int):
    paulis = [pauli for pauli in paulis if pauli is not None]
    qubits = _make_qubits(len(paulis))
    pauli_string = cirq.PauliString(qubit_pauli_map={q: p for q, p in zip(qubits, paulis)}, coefficient=sign)
    actual = cirq.Circuit(cirq.PauliStringPhasor(pauli_string, exponent_neg=phase_exponent_negative)).unitary()
    to_z_mats = {cirq.X: cirq.unitary(cirq.Y ** (-0.5)), cirq.Y: cirq.unitary(cirq.X ** 0.5), cirq.Z: np.eye(2)}
    expected_convert = np.eye(1)
    for pauli in paulis:
        expected_convert = np.kron(expected_convert, to_z_mats[pauli])
    t = 1j ** (phase_exponent_negative * 2 * sign)
    expected_z = np.diag([1, t, t, 1, t, 1, 1, t][:2 ** len(paulis)])
    expected = expected_convert.T.conj().dot(expected_z).dot(expected_convert)
    cirq.testing.assert_allclose_up_to_global_phase(actual, expected, rtol=1e-07, atol=1e-07)