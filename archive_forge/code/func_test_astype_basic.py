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
@pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
@pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
@pytest.mark.filterwarnings('ignore::numpy.ComplexWarning')
def test_astype_basic(dt1, dt2):
    src = np.ma.array(ones(3, dt1), fill_value=1)
    dst = src.astype(dt2)
    assert_(src.fill_value == 1)
    assert_(src.dtype == dt1)
    assert_(src.fill_value.dtype == dt1)
    assert_(dst.fill_value == 1)
    assert_(dst.dtype == dt2)
    assert_(dst.fill_value.dtype == dt2)
    assert_equal(src, dst)