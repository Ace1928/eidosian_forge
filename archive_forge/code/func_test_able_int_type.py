import os
from platform import machine
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..casting import (
from ..testing import suppress_warnings
def test_able_int_type():
    for vals, exp_out in (([0, 1], np.uint8), ([0, 255], np.uint8), ([-1, 1], np.int8), ([0, 256], np.uint16), ([-1, 128], np.int16), ([0.1, 1], None), ([0, 2 ** 16], np.uint32), ([-1, 2 ** 15], np.int32), ([0, 2 ** 32], np.uint64), ([-1, 2 ** 31], np.int64), ([-1, 2 ** 64 - 1], None), ([0, 2 ** 64 - 1], np.uint64), ([0, 2 ** 64], None)):
        assert able_int_type(vals) == exp_out