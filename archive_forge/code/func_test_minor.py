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
@pytest.mark.parametrize('lims, expected_low_ticks', zip(ref_basic_limits, ref_basic_major_ticks))
def test_minor(self, lims, expected_low_ticks):
    """
        In large scale, test the presence of minor,
        and assert no minor when major are subsampled.
        """
    expected_ticks = sorted([*expected_low_ticks, 0.5, *1 - expected_low_ticks])
    basic_needed = len(expected_ticks)
    loc = mticker.LogitLocator(nbins=100)
    minor_loc = mticker.LogitLocator(nbins=100, minor=True)
    for nbins in range(basic_needed, 2, -1):
        loc.set_params(nbins=nbins)
        minor_loc.set_params(nbins=nbins)
        major_ticks = loc.tick_values(*lims)
        minor_ticks = minor_loc.tick_values(*lims)
        if len(major_ticks) >= len(expected_ticks):
            assert (len(major_ticks) - 1) * 5 < len(minor_ticks)
        else:
            _LogitHelper.assert_almost_equal(sorted([*major_ticks, *minor_ticks]), expected_ticks)