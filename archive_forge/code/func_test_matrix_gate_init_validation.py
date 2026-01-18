import re
import numpy as np
import pytest
import sympy
import cirq
def test_matrix_gate_init_validation():
    with pytest.raises(ValueError, match='square 2d numpy array'):
        _ = cirq.MatrixGate(np.ones(shape=(1, 1, 1)))
    with pytest.raises(ValueError, match='square 2d numpy array'):
        _ = cirq.MatrixGate(np.ones(shape=(2, 1)))
    with pytest.raises(ValueError, match='not a power of 2'):
        _ = cirq.MatrixGate(np.ones(shape=(0, 0)))
    with pytest.raises(ValueError, match='not a power of 2'):
        _ = cirq.MatrixGate(np.eye(3))
    with pytest.raises(ValueError, match='matrix shape for qid_shape'):
        _ = cirq.MatrixGate(np.eye(3), qid_shape=(4,))