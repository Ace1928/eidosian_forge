import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
@pytest.mark.parametrize('sparse_tensor_type', [pa.SparseCSRMatrix, pa.SparseCSCMatrix, pa.SparseCOOTensor, pa.SparseCSFTensor])
@pytest.mark.parametrize('dtype_str,arrow_type', tensor_type_pairs)
def test_dense_to_sparse_tensor(dtype_str, arrow_type, sparse_tensor_type):
    dtype = np.dtype(dtype_str)
    array = np.array([[4, 0, 9, 0], [0, 7, 0, 0], [0, 0, 0, 0], [0, 0, 0, 5]]).astype(dtype)
    dim_names = ('x', 'y')
    sparse_tensor = sparse_tensor_type.from_dense_numpy(array, dim_names)
    tensor = sparse_tensor.to_tensor()
    result_array = tensor.to_numpy()
    assert sparse_tensor.type == arrow_type
    assert tensor.type == arrow_type
    assert sparse_tensor.dim_names == dim_names
    assert np.array_equal(array, result_array)