import logging
import numpy as np
import pandas as pd
from param import get_logger
from holoviews.core.data import Dataset, MultiInterface
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
def test_array_values_coordinates_nonexpanded_constant_kdim(self):
    arrays = [np.column_stack([np.arange(i, i + 2), np.arange(i, i + 2), np.ones(2) * i]) for i in range(2)]
    mds = Path(arrays, kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
    self.assertIs(mds.interface, self.interface)
    self.assertEqual(mds.dimension_values(2, expanded=False), np.array([0, 1]))