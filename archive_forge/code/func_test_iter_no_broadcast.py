import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_no_broadcast():
    a = np.arange(24).reshape(2, 3, 4)
    b = np.arange(6).reshape(2, 3, 1)
    c = np.arange(12).reshape(3, 4)
    nditer([a, b, c], [], [['readonly', 'no_broadcast'], ['readonly'], ['readonly']])
    assert_raises(ValueError, nditer, [a, b, c], [], [['readonly'], ['readonly', 'no_broadcast'], ['readonly']])
    assert_raises(ValueError, nditer, [a, b, c], [], [['readonly'], ['readonly'], ['readonly', 'no_broadcast']])