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
def test_cax_legend_height(self):
    """Pass a cax argument to 'df.plot(.)', the legend location must be
        aligned with those of main plot
        """
    with warnings.catch_warnings(record=True) as _:
        ax = self.df.plot(column='pop_est', cmap='OrRd', legend=True)
    plot_height = _get_ax(ax.get_figure(), '').get_position().height
    legend_height = _get_ax(ax.get_figure(), '<colorbar>').get_position().height
    assert abs(plot_height - legend_height) >= 1e-06
    fig, ax2 = plt.subplots()
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.1, label='fixed_colorbar')
    with warnings.catch_warnings(record=True) as _:
        ax2 = self.df.plot(column='pop_est', cmap='OrRd', legend=True, cax=cax, ax=ax2)
    plot_height = _get_ax(fig, '').get_position().height
    legend_height = _get_ax(fig, 'fixed_colorbar').get_position().height
    assert abs(plot_height - legend_height) < 1e-06