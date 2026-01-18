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
def sparse_test_class(getset=True, slicing=True, slicing_assign=True, fancy_indexing=True, fancy_assign=True, fancy_multidim_indexing=True, fancy_multidim_assign=True, minmax=True, nnz_axis=True):
    """
    Construct a base class, optionally converting some of the tests in
    the suite to check that the feature is not implemented.
    """
    bases = (_TestCommon, _possibly_unimplemented(_TestGetSet, getset), _TestSolve, _TestInplaceArithmetic, _TestArithmetic, _possibly_unimplemented(_TestSlicing, slicing), _possibly_unimplemented(_TestSlicingAssign, slicing_assign), _possibly_unimplemented(_TestFancyIndexing, fancy_indexing), _possibly_unimplemented(_TestFancyIndexingAssign, fancy_assign), _possibly_unimplemented(_TestFancyMultidim, fancy_indexing and fancy_multidim_indexing), _possibly_unimplemented(_TestFancyMultidimAssign, fancy_multidim_assign and fancy_assign), _possibly_unimplemented(_TestMinMax, minmax), _possibly_unimplemented(_TestGetNnzAxis, nnz_axis))
    names = {}
    for cls in bases:
        for name in cls.__dict__:
            if not name.startswith('test_'):
                continue
            old_cls = names.get(name)
            if old_cls is not None:
                raise ValueError('Test class {} overloads test {} defined in {}'.format(cls.__name__, name, old_cls.__name__))
            names[name] = cls
    return type('TestBase', bases, {})