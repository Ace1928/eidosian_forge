from typing import cast
import numpy as np
import pytest
import cirq
def test_measurement_channel():
    np.testing.assert_allclose(cirq.kraus(cirq.MeasurementGate(1, 'a')), (np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])))
    cirq.testing.assert_consistent_channel(cirq.MeasurementGate(1, 'a'))
    assert not cirq.has_mixture(cirq.MeasurementGate(1, 'a'))
    np.testing.assert_allclose(cirq.kraus(cirq.MeasurementGate(2, 'a')), (np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]), np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])))
    np.testing.assert_allclose(cirq.kraus(cirq.MeasurementGate(2, 'a', qid_shape=(2, 3))), (np.diag([1, 0, 0, 0, 0, 0]), np.diag([0, 1, 0, 0, 0, 0]), np.diag([0, 0, 1, 0, 0, 0]), np.diag([0, 0, 0, 1, 0, 0]), np.diag([0, 0, 0, 0, 1, 0]), np.diag([0, 0, 0, 0, 0, 1])))