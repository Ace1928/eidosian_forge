import numpy as np
import pytest
import cirq
import cirq.testing
def test_infer_qid_shape():
    computational_basis_state_1 = [0, 0, 0, 1]
    computational_basis_state_2 = [0, 1, 2, 3]
    computational_basis_state_3 = [0, 1, 2, 4]
    computational_basis_state_4 = 9
    computational_basis_state_5 = [0, 1, 2, 4, 5]
    state_vector_1 = cirq.one_hot(shape=(4,), dtype=np.complex64)
    state_vector_2 = cirq.one_hot(shape=(24,), dtype=np.complex64)
    state_tensor_1 = np.reshape(state_vector_1, (2, 2))
    state_tensor_2 = np.reshape(state_vector_2, (1, 2, 3, 4))
    density_matrix_1 = np.eye(4, dtype=np.complex64) / 4
    density_matrix_2 = np.eye(24, dtype=np.complex64) / 24
    q0, q1 = cirq.LineQubit.range(2)
    product_state_1 = cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1)
    assert cirq.qis.infer_qid_shape(computational_basis_state_1, state_vector_1, state_tensor_1, density_matrix_1, product_state_1) == (2, 2)
    assert cirq.qis.infer_qid_shape(product_state_1, density_matrix_1, state_tensor_1, state_vector_1, computational_basis_state_1) == (2, 2)
    assert cirq.qis.infer_qid_shape(computational_basis_state_1, computational_basis_state_2, computational_basis_state_4, state_tensor_2) == (1, 2, 3, 4)
    assert cirq.qis.infer_qid_shape(state_vector_2, density_matrix_2, computational_basis_state_4) == (24,)
    assert cirq.qis.infer_qid_shape(state_tensor_2, density_matrix_2) == (1, 2, 3, 4)
    assert cirq.qis.infer_qid_shape(computational_basis_state_4) == (10,)
    assert cirq.qis.infer_qid_shape(15, 7, 22, 4) == (23,)
    with pytest.raises(ValueError, match='No states were specified'):
        _ = cirq.qis.infer_qid_shape()
    with pytest.raises(ValueError, match='Failed'):
        _ = cirq.qis.infer_qid_shape(computational_basis_state_1, computational_basis_state_5)
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.qis.infer_qid_shape(computational_basis_state_1)
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.qis.infer_qid_shape(state_tensor_1)
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.qis.infer_qid_shape(density_matrix_1)
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.qis.infer_qid_shape(computational_basis_state_1, computational_basis_state_2)
    with pytest.raises(ValueError, match='Failed'):
        _ = cirq.qis.infer_qid_shape(state_vector_1, computational_basis_state_4)
    with pytest.raises(ValueError, match='Failed to infer'):
        _ = cirq.qis.infer_qid_shape(state_vector_1, state_vector_2)
    with pytest.raises(ValueError, match='Failed to infer'):
        _ = cirq.qis.infer_qid_shape(computational_basis_state_3, state_tensor_2)