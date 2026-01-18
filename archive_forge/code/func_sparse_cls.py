import pytest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
from numpy.testing import assert_allclose, assert_equal
@pytest.fixture(params=sparse_params)
def sparse_cls(request):
    return getattr(sparse, request.param)