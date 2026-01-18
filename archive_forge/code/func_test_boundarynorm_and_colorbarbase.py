import copy
import itertools
import unittest.mock
from packaging.version import parse as parse_version
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
import base64
from numpy.testing import assert_array_equal, assert_array_almost_equal
from matplotlib import cbook, cm
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.rcsetup import cycler
from matplotlib.testing.decorators import image_comparison, check_figures_equal
@image_comparison(baseline_images=['boundarynorm_and_colorbar'], extensions=['png'], tol=1.0)
def test_boundarynorm_and_colorbarbase():
    plt.rcParams['pcolormesh.snap'] = False
    fig = plt.figure()
    ax1 = fig.add_axes([0.05, 0.8, 0.9, 0.15])
    ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])
    ax3 = fig.add_axes([0.05, 0.15, 0.9, 0.15])
    bounds = [-1, 2, 5, 7, 12, 15]
    cmap = mpl.colormaps['viridis']
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cb1 = mcolorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, extend='both', orientation='horizontal', spacing='uniform')
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend='both')
    cb2 = mcolorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation='horizontal')
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend='both')
    cb3 = mcolorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, extend='neither', orientation='horizontal')