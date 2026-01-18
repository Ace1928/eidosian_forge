import itertools
import platform
import timeit
from types import SimpleNamespace
from cycler import cycler
import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib
import matplotlib as mpl
from matplotlib import _path
import matplotlib.lines as mlines
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@check_figures_equal(extensions=('png',))
def test_no_subslice_with_transform(fig_ref, fig_test):
    ax = fig_ref.add_subplot()
    x = np.arange(2000)
    ax.plot(x + 2000, x)
    ax = fig_test.add_subplot()
    t = mtransforms.Affine2D().translate(2000.0, 0.0)
    ax.plot(x, x, transform=t + ax.transData)