import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_ctx_mgr_as_smth(self):
    opts = {'precision': 2}
    with np.printoptions(**opts) as ctx:
        saved_opts = ctx.copy()
    assert_equal({k: saved_opts[k] for k in opts}, opts)