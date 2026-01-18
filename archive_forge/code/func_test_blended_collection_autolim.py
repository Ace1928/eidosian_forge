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
def test_blended_collection_autolim():
    a = [1, 2, 4]
    height = 0.2
    xy_pairs = np.column_stack([np.repeat(a, 2), np.tile([0, height], len(a))])
    line_segs = xy_pairs.reshape([len(a), 2, 2])
    f, ax = plt.subplots()
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.add_collection(LineCollection(line_segs, transform=trans))
    ax.autoscale_view(scalex=True, scaley=False)
    np.testing.assert_allclose(ax.get_xlim(), [1.0, 4.0])