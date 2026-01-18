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
def test_minor_number(self):
    """
        Test the parameter minor_number
        """
    min_loc = mticker.LogitLocator(minor=True)
    min_form = mticker.LogitFormatter(minor=True)
    ticks = min_loc.tick_values(0.05, 1 - 0.05)
    for minor_number in (2, 4, 8, 16):
        min_form.set_minor_number(minor_number)
        formatted = min_form.format_ticks(ticks)
        labelled = [f for f in formatted if len(f) > 0]
        assert len(labelled) == minor_number