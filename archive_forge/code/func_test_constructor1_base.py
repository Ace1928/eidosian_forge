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
def test_constructor1_base(self):
    A = self.datsp
    self_format = A.format
    C = A.__class__(A, copy=False)
    assert_array_equal_dtype(A.toarray(), C.toarray())
    if self_format not in NON_ARRAY_BACKED_FORMATS:
        assert_(sparse_may_share_memory(A, C))
    C = A.__class__(A, dtype=A.dtype, copy=False)
    assert_array_equal_dtype(A.toarray(), C.toarray())
    if self_format not in NON_ARRAY_BACKED_FORMATS:
        assert_(sparse_may_share_memory(A, C))
    C = A.__class__(A, dtype=np.float32, copy=False)
    assert_array_equal(A.toarray(), C.toarray())
    C = A.__class__(A, copy=True)
    assert_array_equal_dtype(A.toarray(), C.toarray())
    assert_(not sparse_may_share_memory(A, C))
    for other_format in ['csr', 'csc', 'coo', 'dia', 'dok', 'lil']:
        if other_format == self_format:
            continue
        B = A.asformat(other_format)
        C = A.__class__(B, copy=False)
        assert_array_equal_dtype(A.toarray(), C.toarray())
        C = A.__class__(B, copy=True)
        assert_array_equal_dtype(A.toarray(), C.toarray())
        assert_(not sparse_may_share_memory(B, C))