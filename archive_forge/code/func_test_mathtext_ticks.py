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
def test_mathtext_ticks(self):
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': 'cmr10', 'axes.formatter.use_mathtext': False})
    if parse_version(pytest.__version__).major < 8:
        with pytest.warns(UserWarning, match='cmr10 font should ideally'):
            fig, ax = plt.subplots()
            ax.set_xticks([-1, 0, 1])
            fig.canvas.draw()
    else:
        with pytest.warns(UserWarning, match='Glyph 8722'), pytest.warns(UserWarning, match='cmr10 font should ideally'):
            fig, ax = plt.subplots()
            ax.set_xticks([-1, 0, 1])
            fig.canvas.draw()