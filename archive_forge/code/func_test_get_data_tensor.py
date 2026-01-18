from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
@pytest.mark.parametrize('data', [np.array([[1, 2], [3, 4]]), sp.eye(2) * 4])
def test_get_data_tensor(self, scipy_backend, data):
    outer = scipy_backend.get_data_tensor(data)
    assert outer.keys() == {-1}, 'Should only be constant variable ID.'
    inner = outer[-1]
    assert inner.keys() == {-1}, 'Should only be in parameter slice -1, i.e. non parametrized.'
    tensor = inner[-1]
    assert isinstance(tensor, sp.spmatrix), 'Should be a scipy sparse matrix'
    assert tensor.shape == (4, 1), 'Should be a 1*4x1 tensor'
    expected = sp.csr_matrix(data.reshape((-1, 1), order='F'))
    assert (tensor != expected).nnz == 0