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
def test_linestrings_values(self):
    from geopandas.plotting import _plot_linestring_collection
    fig, ax = plt.subplots()
    coll = _plot_linestring_collection(ax, self.lines, self.values)
    fig.canvas.draw_idle()
    cmap = plt.get_cmap()
    expected_colors = cmap(np.arange(self.N) / (self.N - 1))
    _check_colors(self.N, coll.get_color(), expected_colors)
    ax.cla()
    coll = _plot_linestring_collection(ax, self.lines, self.values, cmap='RdBu')
    fig.canvas.draw_idle()
    cmap = plt.get_cmap('RdBu')
    expected_colors = cmap(np.arange(self.N) / (self.N - 1))
    _check_colors(self.N, coll.get_color(), expected_colors)
    ax.cla()
    coll = _plot_linestring_collection(ax, self.lines, self.values, vmin=3, vmax=5)
    fig.canvas.draw_idle()
    cmap = plt.get_cmap()
    expected_colors = [cmap(0)]
    _check_colors(self.N, coll.get_color(), expected_colors * 3)
    ax.cla()