from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.skipif(compat.USE_SHAPELY_20 or not (compat.USE_PYGEOS and (not compat.PYGEOS_GE_010)), reason='sindex.nearest exclusive parameter requires shapely >= 2.0')
def test_nearest_exclusive_unavailable(self):
    from shapely.geometry import Point
    geoms = [Point((x, y)) for x, y in zip(np.arange(5), np.arange(5))]
    df = geopandas.GeoDataFrame(geometry=geoms)
    with pytest.raises(NotImplementedError, match='requires shapely >= 2.0'):
        df.sindex.nearest(geoms, exclusive=True)