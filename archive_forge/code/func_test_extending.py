from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
def test_extending(self):
    sym = mticker.SymmetricalLogLocator(base=10, linthresh=1)
    sym.create_dummy_axis()
    sym.axis.set_view_interval(8, 9)
    assert (sym() == [1.0]).all()
    sym.axis.set_view_interval(8, 12)
    assert (sym() == [1.0, 10.0]).all()
    assert sym.view_limits(10, 10) == (1, 100)
    assert sym.view_limits(-10, -10) == (-100, -1)
    assert sym.view_limits(0, 0) == (-0.001, 0.001)