import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
import pytest
@pytest.mark.parametrize('matrix_input, axis, expected_shape', [(csc_matrix([[1, 0], [0, 0], [0, 2]]), 0, (0, 2)), (csc_matrix([[1, 0], [0, 0], [0, 2]]), 1, (3, 0)), (csc_matrix([[1, 0], [0, 0], [0, 2]]), 'both', (0, 0)), (csc_matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 2, 3, 0, 1]]), 0, (0, 6))])
def test_csc_empty_slices(matrix_input, axis, expected_shape):
    slice_1 = matrix_input.toarray().shape[0] - 1
    slice_2 = slice_1
    slice_3 = slice_2 - 1
    if axis == 0:
        actual_shape_1 = matrix_input[slice_1:slice_2, :].toarray().shape
        actual_shape_2 = matrix_input[slice_1:slice_3, :].toarray().shape
    elif axis == 1:
        actual_shape_1 = matrix_input[:, slice_1:slice_2].toarray().shape
        actual_shape_2 = matrix_input[:, slice_1:slice_3].toarray().shape
    elif axis == 'both':
        actual_shape_1 = matrix_input[slice_1:slice_2, slice_1:slice_2].toarray().shape
        actual_shape_2 = matrix_input[slice_1:slice_3, slice_1:slice_3].toarray().shape
    assert actual_shape_1 == expected_shape
    assert actual_shape_1 == actual_shape_2