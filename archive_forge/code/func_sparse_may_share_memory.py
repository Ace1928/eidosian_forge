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
def sparse_may_share_memory(A, B):

    def _underlying_arrays(x):
        arrays = []
        for a in x.__dict__.values():
            if isinstance(a, (np.ndarray, np.generic)):
                arrays.append(a)
        return arrays
    for a in _underlying_arrays(A):
        for b in _underlying_arrays(B):
            if np.may_share_memory(a, b):
                return True
    return False