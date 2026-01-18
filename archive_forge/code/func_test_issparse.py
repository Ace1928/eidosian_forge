import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
def test_issparse():
    m = scipy.sparse.eye(3)
    a = scipy.sparse.csr_array(m)
    assert not isinstance(m, scipy.sparse.sparray)
    assert isinstance(a, scipy.sparse.sparray)
    assert scipy.sparse.issparse(a)
    assert scipy.sparse.issparse(m)
    assert not scipy.sparse.issparse(a.todense())
    assert not scipy.sparse.issparse(m.todense())