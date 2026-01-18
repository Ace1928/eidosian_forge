import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_box_aspect_constructor_rotated(self):
    box = Box(0, 0, 1, aspect=2, orientation=np.pi / 8)
    self.assertEqual(np.allclose(box.data[0], self.rotated_rect), True)