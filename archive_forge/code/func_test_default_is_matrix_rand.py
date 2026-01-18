import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
def test_default_is_matrix_rand():
    m = scipy.sparse.rand(3, 3)
    assert not isinstance(m, scipy.sparse.sparray)