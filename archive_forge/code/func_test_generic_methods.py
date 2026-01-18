import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def test_generic_methods(self):
    a = array([1, 3, 2])
    assert_equal(a.any(), a._data.any())
    assert_equal(a.all(), a._data.all())
    assert_equal(a.argmax(), a._data.argmax())
    assert_equal(a.argmin(), a._data.argmin())
    assert_equal(a.choose(0, 1, 2, 3, 4), a._data.choose(0, 1, 2, 3, 4))
    assert_equal(a.compress([1, 0, 1]), a._data.compress([1, 0, 1]))
    assert_equal(a.conj(), a._data.conj())
    assert_equal(a.conjugate(), a._data.conjugate())
    m = array([[1, 2], [3, 4]])
    assert_equal(m.diagonal(), m._data.diagonal())
    assert_equal(a.sum(), a._data.sum())
    assert_equal(a.take([1, 2]), a._data.take([1, 2]))
    assert_equal(m.transpose(), m._data.transpose())