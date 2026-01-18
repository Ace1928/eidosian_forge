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
@pytest.mark.parametrize('copy', [True, False])
def test_masked_invalid_full_mask(self, copy):
    a = np.ma.array([1, 2, 3, 4])
    assert a._mask is nomask
    res = np.ma.masked_invalid(a, copy=copy)
    assert res.mask is not nomask
    assert a.mask is nomask
    assert np.may_share_memory(a._data, res._data) != copy