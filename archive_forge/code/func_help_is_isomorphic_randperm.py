import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_, assert_warns
import pytest
from pytest import raises as assert_raises
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import (
from scipy.spatial.distance import pdist
from scipy.cluster._hierarchy import Heap
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close
from . import hierarchy_test_data
def help_is_isomorphic_randperm(self, nobs, nclusters, noniso=False, nerrors=0):
    for k in range(3):
        a = (np.random.rand(nobs) * nclusters).astype(int)
        b = np.zeros(a.size, dtype=int)
        P = np.random.permutation(nclusters)
        for i in range(0, a.shape[0]):
            b[i] = P[a[i]]
        if noniso:
            Q = np.random.permutation(nobs)
            b[Q[0:nerrors]] += 1
            b[Q[0:nerrors]] %= nclusters
        assert_(is_isomorphic(a, b) == (not noniso))
        assert_(is_isomorphic(b, a) == (not noniso))