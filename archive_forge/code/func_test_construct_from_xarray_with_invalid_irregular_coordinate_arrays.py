from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_construct_from_xarray_with_invalid_irregular_coordinate_arrays(self):
    try:
        import xarray as xr
    except ImportError:
        raise SkipTest('Test requires xarray')
    zs = np.arange(48 * 6).reshape(2, 4, 6, 6)
    da = xr.DataArray(zs, dims=['z', 'y', 'x', 'b'], coords={'lat': (('y', 'b'), self.ys), 'lon': (('y', 'x'), self.xs), 'z': [0, 1]}, name='A')
    with self.assertRaises(DataError):
        Dataset(da, ['z', 'lon', 'lat'])