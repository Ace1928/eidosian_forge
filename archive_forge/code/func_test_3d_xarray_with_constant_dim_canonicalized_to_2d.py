from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_3d_xarray_with_constant_dim_canonicalized_to_2d(self):
    try:
        import xarray as xr
    except ImportError:
        raise SkipTest('Test requires xarray')
    zs = np.arange(24).reshape(1, 4, 6)
    da = xr.DataArray(zs, dims=['z', 'y', 'x'], coords={'lat': (('y', 'x'), self.ys), 'lon': (('y', 'x'), self.xs), 'z': [0]}, name='A')
    dataset = Dataset(da, ['lon', 'lat'], 'A')
    self.assertEqual(dataset.dimension_values('A', flat=False), zs[0])