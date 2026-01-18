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
def check_2(a, b):
    if isinstance(a, np.ndarray):
        ai = int(a)
    else:
        ai = a
    if isinstance(b, np.ndarray):
        bi = int(b)
    else:
        bi = b
    x = A[a, b]
    y = B[ai, bi]
    if y.shape == ():
        assert_equal(x, y, repr((a, b)))
    elif x.size == 0 and y.size == 0:
        pass
    else:
        assert_array_equal(x.toarray(), y, repr((a, b)))