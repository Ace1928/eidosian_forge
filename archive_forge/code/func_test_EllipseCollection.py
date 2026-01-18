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
@image_comparison(['EllipseCollection_test_image.png'], remove_text=True)
def test_EllipseCollection():
    fig, ax = plt.subplots()
    x = np.arange(4)
    y = np.arange(3)
    X, Y = np.meshgrid(x, y)
    XY = np.vstack((X.ravel(), Y.ravel())).T
    ww = X / x[-1]
    hh = Y / y[-1]
    aa = np.ones_like(ww) * 20
    ec = mcollections.EllipseCollection(ww, hh, aa, units='x', offsets=XY, offset_transform=ax.transData, facecolors='none')
    ax.add_collection(ec)
    ax.autoscale_view()