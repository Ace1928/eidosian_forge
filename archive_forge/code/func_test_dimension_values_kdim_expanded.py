from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset, concat
from geoviews.data.iris import coord_to_dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Image
from holoviews.tests.core.data.test_imageinterface import BaseImageElementInterfaceTests
from holoviews.tests.core.data.test_gridinterface import BaseGridInterfaceTests
def test_dimension_values_kdim_expanded(self):
    cube = Dataset(self.cube, kdims=['longitude', 'latitude'])
    self.assertEqual(cube.dimension_values('longitude'), np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32))