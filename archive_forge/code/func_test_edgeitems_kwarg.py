import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_edgeitems_kwarg(self):
    arr = np.zeros(3, int)
    assert_equal(np.array2string(arr, edgeitems=1, threshold=0), '[0 ... 0]')