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
def test_legend(self):
    with warnings.catch_warnings(record=True) as _:
        ax = self.df.plot(column='pop_est', scheme='QUANTILES', k=3, cmap='OrRd', legend=True)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    expected = [s.split('|')[0][1:-2] for s in str(self.mc.Quantiles(self.df['pop_est'], k=3)).split('\n')[4:]]
    assert labels == expected