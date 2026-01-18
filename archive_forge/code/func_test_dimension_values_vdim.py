from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset, concat
from geoviews.data.iris import coord_to_dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Image
from holoviews.tests.core.data.test_imageinterface import BaseImageElementInterfaceTests
from holoviews.tests.core.data.test_gridinterface import BaseGridInterfaceTests
def test_dimension_values_vdim(self):
    cube = Dataset(self.cube, kdims=['longitude', 'latitude'])
    self.assertEqual(cube.dimension_values('unknown', flat=False), np.array([[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]], dtype=np.int32).T)