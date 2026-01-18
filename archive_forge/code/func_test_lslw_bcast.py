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
@mpl.style.context('default')
def test_lslw_bcast():
    col = mcollections.PathCollection([])
    col.set_linestyles(['-', '-'])
    col.set_linewidths([1, 2, 3])
    assert col.get_linestyles() == [(0, None)] * 6
    assert col.get_linewidths() == [1, 2, 3] * 2
    col.set_linestyles(['-', '-', '-'])
    assert col.get_linestyles() == [(0, None)] * 3
    assert (col.get_linewidths() == [1, 2, 3]).all()