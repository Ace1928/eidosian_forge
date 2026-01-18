import pytest
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy import sparse
from scipy.sparse import csgraph
from scipy._lib._util import np_long, np_ulong
def test_format_error_message():
    with pytest.raises(ValueError, match="Invalid form: 'toto'"):
        _ = csgraph.laplacian(np.eye(1), form='toto')