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
@pytest.mark.parametrize('data, expected', cursor_data)
def test_cursor_precision(self, data, expected):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    fmt = ax.xaxis.get_major_formatter().format_data_short
    assert fmt(data) == expected