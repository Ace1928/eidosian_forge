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
def test_near_zero(self):
    """Check that manually injected zero will supersede nearby tick"""
    lctr = mticker.AsinhLocator(linear_width=100, numticks=3, base=0)
    assert_almost_equal(lctr.tick_values(-1.1, 0.9), [-1.0, 0.0, 0.9])