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
@check_figures_equal()
def test_input_copy(fig_test, fig_ref):
    t = np.arange(0, 6, 2)
    l, = fig_test.add_subplot().plot(t, t, '.-')
    t[:] = range(3)
    l.set_drawstyle('steps')
    fig_ref.add_subplot().plot([0, 2, 4], [0, 2, 4], '.-', drawstyle='steps')