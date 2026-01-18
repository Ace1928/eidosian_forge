import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_format_function(self):
    """Test custom format function for each element in array."""

    def _format_function(x):
        if np.abs(x) < 1:
            return '.'
        elif np.abs(x) < 2:
            return 'o'
        else:
            return 'O'
    x = np.arange(3)
    x_hex = '[0x0 0x1 0x2]'
    x_oct = '[0o0 0o1 0o2]'
    assert_(np.array2string(x, formatter={'all': _format_function}) == '[. o O]')
    assert_(np.array2string(x, formatter={'int_kind': _format_function}) == '[. o O]')
    assert_(np.array2string(x, formatter={'all': lambda x: '%.4f' % x}) == '[0.0000 1.0000 2.0000]')
    assert_equal(np.array2string(x, formatter={'int': lambda x: hex(x)}), x_hex)
    assert_equal(np.array2string(x, formatter={'int': lambda x: oct(x)}), x_oct)
    x = np.arange(3.0)
    assert_(np.array2string(x, formatter={'float_kind': lambda x: '%.2f' % x}) == '[0.00 1.00 2.00]')
    assert_(np.array2string(x, formatter={'float': lambda x: '%.2f' % x}) == '[0.00 1.00 2.00]')
    s = np.array(['abc', 'def'])
    assert_(np.array2string(s, formatter={'numpystr': lambda s: s * 2}) == '[abcabc defdef]')