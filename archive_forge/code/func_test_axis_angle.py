import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
def test_axis_angle():
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.ry(1e-10))), cirq.AxisAngleDecomposition(angle=0, axis=(1, 0, 0), global_phase=1), atol=1e-08)
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.rx(np.pi))), cirq.AxisAngleDecomposition(angle=np.pi, axis=(1, 0, 0), global_phase=1), atol=1e-08)
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.X)), cirq.AxisAngleDecomposition(angle=np.pi, axis=(1, 0, 0), global_phase=1j), atol=1e-08)
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.X ** 0.5)), cirq.AxisAngleDecomposition(angle=np.pi / 2, axis=(1, 0, 0), global_phase=np.exp(1j * np.pi / 4)), atol=1e-08)
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.X ** (-0.5))), cirq.AxisAngleDecomposition(angle=-np.pi / 2, axis=(1, 0, 0), global_phase=np.exp(-1j * np.pi / 4)))
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.Y)), cirq.AxisAngleDecomposition(angle=np.pi, axis=(0, 1, 0), global_phase=1j), atol=1e-08)
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.Z)), cirq.AxisAngleDecomposition(angle=np.pi, axis=(0, 0, 1), global_phase=1j), atol=1e-08)
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.H)), cirq.AxisAngleDecomposition(angle=np.pi, axis=(np.sqrt(0.5), 0, np.sqrt(0.5)), global_phase=1j), atol=1e-08)
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.H ** 0.5)), cirq.AxisAngleDecomposition(angle=np.pi / 2, axis=(np.sqrt(0.5), 0, np.sqrt(0.5)), global_phase=np.exp(1j * np.pi / 4)), atol=1e-08)