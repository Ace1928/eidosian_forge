import numpy as np
import pytest
import cirq
import cirq.testing
def assert_valid_density_matrix(matrix, num_qubits=None, qid_shape=None):
    if qid_shape is None and num_qubits is None:
        num_qubits = 1
    np.testing.assert_almost_equal(cirq.to_valid_density_matrix(matrix, num_qubits=num_qubits, qid_shape=qid_shape, dtype=matrix.dtype), matrix)