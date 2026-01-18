import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_nditer_multi_index_set():
    a = np.arange(6).reshape(2, 3)
    it = np.nditer(a, flags=['multi_index'])
    it.multi_index = (0, 2)
    assert_equal([i for i in it], [2, 3, 4, 5])