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
def test_LineCollection_args():
    lc = LineCollection(None, linewidth=2.2, edgecolor='r', zorder=3, facecolors=[0, 1, 0, 1])
    assert lc.get_linewidth()[0] == 2.2
    assert mcolors.same_color(lc.get_edgecolor(), 'r')
    assert lc.get_zorder() == 3
    assert mcolors.same_color(lc.get_facecolor(), [[0, 1, 0, 1]])
    lc = LineCollection(None, facecolor=None)
    assert mcolors.same_color(lc.get_facecolor(), 'none')