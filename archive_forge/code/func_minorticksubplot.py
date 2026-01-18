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
def minorticksubplot(xminor, yminor, i):
    rc = {'xtick.minor.visible': xminor, 'ytick.minor.visible': yminor}
    with plt.rc_context(rc=rc):
        ax = fig.add_subplot(2, 2, i)
    assert (len(ax.xaxis.get_minor_ticks()) > 0) == xminor
    assert (len(ax.yaxis.get_minor_ticks()) > 0) == yminor