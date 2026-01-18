import numpy as np
from holoviews.element import HSV, RGB, Curve, Image, QuadMesh, Raster
from holoviews.element.comparison import ComparisonTestCase
def test_cast_image_to_quadmesh(self):
    img = Image(self.array1, kdims=['a', 'b'], vdims=['c'], group='A', label='B')
    qmesh = QuadMesh(img)
    self.assertEqual(qmesh.dimension_values(0, False), np.array([-0.333333, 0.0, 0.333333]))
    self.assertEqual(qmesh.dimension_values(1, False), np.array([-0.25, 0.25]))
    self.assertEqual(qmesh.dimension_values(2, flat=False), self.array1[::-1])
    self.assertEqual(qmesh.kdims, img.kdims)
    self.assertEqual(qmesh.vdims, img.vdims)
    self.assertEqual(qmesh.group, img.group)
    self.assertEqual(qmesh.label, img.label)