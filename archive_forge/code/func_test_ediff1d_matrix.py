import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_ediff1d_matrix():
    assert isinstance(np.ediff1d(np.matrix(1)), np.matrix)
    assert isinstance(np.ediff1d(np.matrix(1), to_begin=1), np.matrix)