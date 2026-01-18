import numpy as np
from holoviews.element import HSV, RGB, Curve, Image, QuadMesh, Raster
from holoviews.element.comparison import ComparisonTestCase
def test_construct_from_dict_with_alpha(self):
    rgb = RGB({'x': [1, 2, 3], 'y': [1, 2, 3], ('R', 'G', 'B', 'A'): self.rgb_array})
    self.assertEqual(len(rgb.vdims), 4)