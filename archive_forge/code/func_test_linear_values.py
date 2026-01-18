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
def test_linear_values(self):
    lctr = mticker.AsinhLocator(linear_width=100, numticks=11, base=0)
    assert_almost_equal(lctr.tick_values(-1, 1), np.arange(-1, 1.01, 0.2))
    assert_almost_equal(lctr.tick_values(-0.1, 0.1), np.arange(-0.1, 0.101, 0.02))
    assert_almost_equal(lctr.tick_values(-0.01, 0.01), np.arange(-0.01, 0.0101, 0.002))