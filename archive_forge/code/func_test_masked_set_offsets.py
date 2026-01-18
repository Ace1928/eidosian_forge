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
@check_figures_equal(extensions=['png'])
def test_masked_set_offsets(fig_ref, fig_test):
    x = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 1, 0])
    y = np.arange(1, 6)
    ax_test = fig_test.add_subplot()
    scat = ax_test.scatter(x, y)
    scat.set_offsets(np.ma.column_stack([x, y]))
    ax_test.set_xticks([])
    ax_test.set_yticks([])
    ax_ref = fig_ref.add_subplot()
    ax_ref.scatter([1, 2, 5], [1, 2, 5])
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])