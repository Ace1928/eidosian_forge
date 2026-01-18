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
def test_markevery_figure_line_unsupported_relsize():
    fig = plt.figure()
    fig.add_artist(mlines.Line2D([0, 1], [0, 1], marker='o', markevery=0.5))
    with pytest.raises(ValueError):
        fig.canvas.draw()