import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_int64_dtype(self):
    """Check that int64 integer types can be specified"""
    converter = StringConverter(np.int64, default=0)
    val = '-9223372036854775807'
    assert_(converter(val) == -9223372036854775807)
    val = '9223372036854775807'
    assert_(converter(val) == 9223372036854775807)