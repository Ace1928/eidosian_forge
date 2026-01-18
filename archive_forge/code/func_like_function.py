import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def like_function():
    a = np.matrix([[1, 2], [3, 4]])
    for like_function in (np.zeros_like, np.ones_like, np.empty_like):
        b = like_function(a)
        assert_(type(b) is np.matrix)
        c = like_function(a, subok=False)
        assert_(type(c) is not np.matrix)