import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_build_err_msg_custom_precision(self):
    x = np.array([1.000000001, 2.00002, 3.00003])
    y = np.array([1.000000002, 2.00003, 3.00004])
    err_msg = 'There is a mismatch'
    a = build_err_msg([x, y], err_msg, precision=10)
    b = '\nItems are not equal: There is a mismatch\n ACTUAL: array([1.000000001, 2.00002    , 3.00003    ])\n DESIRED: array([1.000000002, 2.00003    , 3.00004    ])'
    assert_equal(a, b)