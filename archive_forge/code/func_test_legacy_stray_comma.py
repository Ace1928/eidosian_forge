import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_legacy_stray_comma(self):
    np.set_printoptions(legacy='1.13')
    assert_equal(str(np.arange(10000)), '[   0    1    2 ..., 9997 9998 9999]')
    np.set_printoptions(legacy=False)
    assert_equal(str(np.arange(10000)), '[   0    1    2 ... 9997 9998 9999]')