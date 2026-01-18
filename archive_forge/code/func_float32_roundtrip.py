import code
import platform
import pytest
import sys
from tempfile import TemporaryFile
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_MUSL
def float32_roundtrip(self):
    x = np.float32(1024 - 2 ** (-14))
    y = np.float32(1024 - 2 ** (-13))
    assert_(repr(x) != repr(y))
    assert_equal(np.float32(repr(x)), x)
    assert_equal(np.float32(repr(y)), y)