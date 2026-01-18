import itertools
from packaging.version import Version
import warnings
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.affinity import rotate
from shapely.geometry import (
from geopandas import GeoDataFrame, GeoSeries, read_file
from geopandas.datasets import get_path
import geopandas._compat as compat
from geopandas.plotting import GeoplotAccessor
import pytest
import matplotlib.pyplot as plt
def test_multipolygons_color(self):
    ax = self.df2.plot()
    assert len(ax.collections[0].get_paths()) == 4
    _check_colors(4, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR] * 4)
    ax = self.df2.plot('values')
    cmap = plt.get_cmap(lut=2)
    expected_colors = [cmap(0), cmap(0), cmap(1), cmap(1)]
    _check_colors(4, ax.collections[0].get_facecolors(), expected_colors)
    ax = self.df2.plot(color=['r', 'b'])
    _check_colors(4, ax.collections[0].get_facecolors(), ['r', 'r', 'b', 'b'])