import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
def dense_setdiag(a, v, k):
    v = np.asarray(v)
    if k >= 0:
        n = min(a.shape[0], a.shape[1] - k)
        if v.ndim != 0:
            n = min(n, len(v))
            v = v[:n]
        i = np.arange(0, n)
        j = np.arange(k, k + n)
        a[i, j] = v
    elif k < 0:
        dense_setdiag(a.T, v, -k)