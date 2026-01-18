from datetime import datetime
import io
import itertools
import re
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.collections import (Collection, LineCollection,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@mpl.style.context('mpl20')
@check_figures_equal(extensions=['png'])
def test_collection_log_datalim(fig_test, fig_ref):
    x_vals = [4.38462e-06, 5.54929e-06, 7.02332e-06, 8.88889e-06, 1.125e-05, 1.42383e-05, 1.80203e-05, 2.2807e-05, 2.88651e-05, 3.65324e-05, 4.62363e-05, 5.85178e-05, 7.40616e-05, 9.37342e-05, 0.000118632]
    y_vals = [0.0, 0.1, 0.182, 0.332, 0.604, 1.1, 2.0, 3.64, 6.64, 12.1, 22.0, 39.6, 71.3]
    x, y = np.meshgrid(x_vals, y_vals)
    x = x.flatten()
    y = y.flatten()
    ax_test = fig_test.subplots()
    ax_test.set_xscale('log')
    ax_test.set_yscale('log')
    ax_test.margins = 0
    ax_test.scatter(x, y)
    ax_ref = fig_ref.subplots()
    ax_ref.set_xscale('log')
    ax_ref.set_yscale('log')
    ax_ref.plot(x, y, marker='o', ls='')