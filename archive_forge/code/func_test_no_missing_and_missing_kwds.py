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
def test_no_missing_and_missing_kwds(self):
    df = self.df.copy()
    df['category'] = df['values'].astype('str')
    df.plot('category', missing_kwds={'facecolor': 'none'}, legend=True)