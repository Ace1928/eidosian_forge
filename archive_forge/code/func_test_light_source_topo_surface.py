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
@image_comparison(['light_source_shading_topo.png'])
def test_light_source_topo_surface():
    """Shades a DEM using different v.e.'s and blend modes."""
    dem = cbook.get_sample_data('jacksboro_fault_dem.npz')
    elev = dem['elevation']
    dx, dy = (dem['dx'], dem['dy'])
    dx = 111320.0 * dx * np.cos(dem['ymin'])
    dy = 111320.0 * dy
    ls = mcolors.LightSource(315, 45)
    cmap = cm.gist_earth
    fig, axs = plt.subplots(nrows=3, ncols=3)
    for row, mode in zip(axs, ['hsv', 'overlay', 'soft']):
        for ax, ve in zip(row, [0.1, 1, 10]):
            rgb = ls.shade(elev, cmap, vert_exag=ve, dx=dx, dy=dy, blend_mode=mode)
            ax.imshow(rgb)
            ax.set(xticks=[], yticks=[])