import numpy as np
import pytest
import sympy
import cirq
def test_fsim_unitary():
    np.testing.assert_allclose(cirq.unitary(cirq.FSimGate(theta=0, phi=0)), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), atol=1e-08)
    np.testing.assert_allclose(cirq.unitary(cirq.FSimGate(theta=np.pi / 2, phi=0)), np.array([[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]]), atol=1e-08)
    np.testing.assert_allclose(cirq.unitary(cirq.FSimGate(theta=-np.pi / 2, phi=0)), np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]), atol=1e-08)
    np.testing.assert_allclose(cirq.unitary(cirq.FSimGate(theta=np.pi, phi=0)), np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]), atol=1e-08)
    np.testing.assert_allclose(cirq.unitary(cirq.FSimGate(theta=2 * np.pi, phi=0)), cirq.unitary(cirq.FSimGate(theta=0, phi=0)), atol=1e-08)
    np.testing.assert_allclose(cirq.unitary(cirq.FSimGate(theta=-np.pi / 2, phi=0)), cirq.unitary(cirq.FSimGate(theta=3 / 2 * np.pi, phi=0)), atol=1e-08)
    np.testing.assert_allclose(cirq.unitary(cirq.FSimGate(theta=0, phi=np.pi / 2)), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1j]]), atol=1e-08)
    np.testing.assert_allclose(cirq.unitary(cirq.FSimGate(theta=0, phi=-np.pi / 2)), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]]), atol=1e-08)
    np.testing.assert_allclose(cirq.unitary(cirq.FSimGate(theta=0, phi=np.pi)), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]), atol=1e-08)
    np.testing.assert_allclose(cirq.unitary(cirq.FSimGate(theta=0, phi=0)), cirq.unitary(cirq.FSimGate(theta=0, phi=2 * np.pi)), atol=1e-08)
    np.testing.assert_allclose(cirq.unitary(cirq.FSimGate(theta=0, phi=-np.pi / 2)), cirq.unitary(cirq.FSimGate(theta=0, phi=3 / 2 * np.pi)), atol=1e-08)
    s = np.sqrt(0.5)
    np.testing.assert_allclose(cirq.unitary(cirq.FSimGate(theta=np.pi / 4, phi=np.pi / 3)), np.array([[1, 0, 0, 0], [0, s, -1j * s, 0], [0, -1j * s, s, 0], [0, 0, 0, 0.5 - 1j * np.sqrt(0.75)]]), atol=1e-08)