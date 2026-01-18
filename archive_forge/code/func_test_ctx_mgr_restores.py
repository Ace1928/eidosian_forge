import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_ctx_mgr_restores(self):
    opts = np.get_printoptions()
    with np.printoptions(precision=opts['precision'] - 1, linewidth=opts['linewidth'] - 4):
        pass
    assert_equal(np.get_printoptions(), opts)