from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_lazy_build(self):
    assert self.df.geometry.values._sindex is None
    assert self.df.sindex.size == 5
    assert self.df.geometry.values._sindex is not None