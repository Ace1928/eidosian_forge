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
def test_multipolygons_alpha(self):
    ax = self.df2.plot(alpha=0.7)
    np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())
    try:
        ax = self.df2.plot(alpha=[0.7, 0.2])
    except TypeError:
        pass
    else:
        np.testing.assert_array_equal([0.7, 0.7, 0.2, 0.2], ax.collections[0].get_alpha())