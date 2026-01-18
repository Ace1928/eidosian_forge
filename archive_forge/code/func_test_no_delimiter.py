import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_no_delimiter(self):
    """Test LineSplitter w/o delimiter"""
    strg = ' 1 2 3 4  5 # test'
    test = LineSplitter()(strg)
    assert_equal(test, ['1', '2', '3', '4', '5'])
    test = LineSplitter('')(strg)
    assert_equal(test, ['1', '2', '3', '4', '5'])