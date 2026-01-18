from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_groupby_3d_from_xarray(self):
    try:
        import xarray as xr
    except ImportError:
        raise SkipTest('Test requires xarray')
    zs = np.arange(48).reshape(2, 4, 6)
    da = xr.DataArray(zs, dims=['z', 'y', 'x'], coords={'lat': (('y', 'x'), self.ys), 'lon': (('y', 'x'), self.xs), 'z': [0, 1]}, name='A')
    grouped = Dataset(da, ['lon', 'lat', 'z'], 'A').groupby('z')
    hmap = HoloMap({0: Dataset((self.xs, self.ys, zs[0]), ['lon', 'lat'], 'A'), 1: Dataset((self.xs, self.ys, zs[1]), ['lon', 'lat'], 'A')}, kdims='z')
    self.assertEqual(grouped, hmap)