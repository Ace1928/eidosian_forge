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
@check_figures_equal(extensions=['png', 'pdf'])
def test_geo_kind(self, fig_test, fig_ref):
    """Test Geo kind."""
    ax1 = fig_test.subplots()
    self.gdf.plot(ax=ax1)
    ax2 = fig_ref.subplots()
    getattr(self.gdf.plot, 'geo')(ax=ax2)
    plt.close('all')