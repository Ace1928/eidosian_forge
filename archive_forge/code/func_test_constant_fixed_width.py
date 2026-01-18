import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_constant_fixed_width(self):
    """Test LineSplitter w/ fixed-width fields"""
    strg = '  1  2  3  4     5   # test'
    test = LineSplitter(3)(strg)
    assert_equal(test, ['1', '2', '3', '4', '', '5', ''])
    strg = '  1     3  4  5  6# test'
    test = LineSplitter(20)(strg)
    assert_equal(test, ['1     3  4  5  6'])
    strg = '  1     3  4  5  6# test'
    test = LineSplitter(30)(strg)
    assert_equal(test, ['1     3  4  5  6'])