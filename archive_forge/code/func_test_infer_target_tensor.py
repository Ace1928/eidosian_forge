from typing import cast, Type
from unittest import mock
import numpy as np
import pytest
import cirq
def test_infer_target_tensor():
    dtype = np.complex64
    args = cirq.StateVectorSimulationState(qubits=cirq.LineQubit.range(2), initial_state=np.array([1.0, 0.0, 0.0, 0.0], dtype=dtype), dtype=dtype)
    np.testing.assert_almost_equal(args.target_tensor, np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j]], dtype=dtype))
    args = cirq.StateVectorSimulationState(qubits=cirq.LineQubit.range(2), initial_state=0, dtype=dtype)
    np.testing.assert_almost_equal(args.target_tensor, np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j]], dtype=dtype))