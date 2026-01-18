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
def test_creation_maskcreation(self):
    data = arange(24, dtype=float)
    data[[3, 6, 15]] = masked
    dma_1 = MaskedArray(data)
    assert_equal(dma_1.mask, data.mask)
    dma_2 = MaskedArray(dma_1)
    assert_equal(dma_2.mask, dma_1.mask)
    dma_3 = MaskedArray(dma_1, mask=[1, 0, 0, 0] * 6)
    fail_if_equal(dma_3.mask, dma_1.mask)
    x = array([1, 2, 3], mask=True)
    assert_equal(x._mask, [True, True, True])
    x = array([1, 2, 3], mask=False)
    assert_equal(x._mask, [False, False, False])
    y = array([1, 2, 3], mask=x._mask, copy=False)
    assert_(np.may_share_memory(x.mask, y.mask))
    y = array([1, 2, 3], mask=x._mask, copy=True)
    assert_(not np.may_share_memory(x.mask, y.mask))
    x = array([1, 2, 3], mask=None)
    assert_equal(x._mask, [False, False, False])