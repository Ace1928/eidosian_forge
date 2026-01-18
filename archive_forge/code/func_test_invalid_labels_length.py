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
def test_invalid_labels_length(self):
    with pytest.raises(ValueError):
        self.df.plot(column='pop_est', scheme='QUANTILES', k=3, cmap='OrRd', legend=True, legend_kwds={'labels': ['foo', 'bar']})