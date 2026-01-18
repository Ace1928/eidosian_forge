import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_canonical_vdim(self):
    x = np.array([0.0, 0.75, 1.5])
    y = np.array([1.5, 0.75, 0.0])
    z = np.array([[0.06925999, 0.05800389, 0.05620127], [0.06240918, 0.05800931, 0.04969735], [0.05376789, 0.04669417, 0.03880118]])
    dataset = self.element((x, y, z), kdims=['x', 'y'], vdims=['z'])
    canonical = np.array([[0.05376789, 0.04669417, 0.03880118], [0.06240918, 0.05800931, 0.04969735], [0.06925999, 0.05800389, 0.05620127]])
    self.assertEqual(dataset.dimension_values('z', flat=False), canonical)