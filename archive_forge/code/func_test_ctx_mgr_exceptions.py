import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_ctx_mgr_exceptions(self):
    opts = np.get_printoptions()
    try:
        with np.printoptions(precision=2, linewidth=11):
            raise ValueError
    except ValueError:
        pass
    assert_equal(np.get_printoptions(), opts)