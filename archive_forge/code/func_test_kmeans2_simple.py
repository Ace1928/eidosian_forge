import warnings
import sys
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.cluster.vq import (kmeans, kmeans2, py_vq, vq, whiten,
from scipy.cluster import _vq
from scipy.conftest import (
from scipy.sparse._sputils import matrix
from scipy._lib._array_api import (
@skip_if_array_api_gpu
@array_api_compatible
def test_kmeans2_simple(self, xp):
    np.random.seed(12345678)
    initc = xp.asarray(np.concatenate([[X[0]], [X[1]], [X[2]]]))
    arrays = [xp.asarray] if SCIPY_ARRAY_API else [np.asarray, matrix]
    for tp in arrays:
        code1 = kmeans2(tp(X), tp(initc), iter=1)[0]
        code2 = kmeans2(tp(X), tp(initc), iter=2)[0]
        xp_assert_close(code1, xp.asarray(CODET1))
        xp_assert_close(code2, xp.asarray(CODET2))