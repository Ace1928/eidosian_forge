import logging
import numpy as np
import pandas as pd
from param import get_logger
from holoviews.core.data import Dataset, MultiInterface
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
def test_dict_groupby_non_scalar(self):
    arrays = [{'x': np.arange(i, i + 2), 'y': i} for i in range(2)]
    mds = Dataset(arrays, kdims=['x', 'y'], datatype=[self.datatype])
    self.assertIs(mds.interface, self.interface)
    with self.assertRaises(ValueError):
        mds.groupby('x')