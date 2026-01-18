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
@pytest.mark.parametrize('lims', ref_maxn_limits)
def test_maxn_major(self, lims):
    """
        When the axis is zoomed, the locator must have the same behavior as
        MaxNLocator.
        """
    loc = mticker.LogitLocator(nbins=100)
    maxn_loc = mticker.MaxNLocator(nbins=100, steps=[1, 2, 5, 10])
    for nbins in (4, 8, 16):
        loc.set_params(nbins=nbins)
        maxn_loc.set_params(nbins=nbins)
        ticks = loc.tick_values(*lims)
        maxn_ticks = maxn_loc.tick_values(*lims)
        assert ticks.shape == maxn_ticks.shape
        assert (ticks == maxn_ticks).all()