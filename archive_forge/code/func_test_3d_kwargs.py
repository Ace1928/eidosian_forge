import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_3d_kwargs(self):
    a = arange(12).reshape(2, 2, 3)

    def myfunc(b, offset=0):
        return b[1 + offset]
    xa = apply_along_axis(myfunc, 2, a, offset=1)
    assert_equal(xa, [[2, 5], [8, 11]])