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
@pytest.mark.parametrize('transform, expected', [('transData', (-0.5, 3.5)), ('transAxes', (2.8, 3.2))])
def test_autolim_with_zeros(transform, expected):
    fig, ax = plt.subplots()
    ax.scatter(0, 0, transform=getattr(ax, transform))
    ax.scatter(3, 3)
    np.testing.assert_allclose(ax.get_ylim(), expected)
    np.testing.assert_allclose(ax.get_xlim(), expected)