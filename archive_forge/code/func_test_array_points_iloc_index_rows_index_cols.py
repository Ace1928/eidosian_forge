from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import MultiBaseInterfaceTest
from geoviews.data.geom_dict import GeomDictInterface
def test_array_points_iloc_index_rows_index_cols(self):
    arrays = [np.array([(1 + i, i), (2 + i, i), (3 + i, i)]) for i in range(2)]
    mds = Dataset(arrays, kdims=['x', 'y'], datatype=[self.datatype])
    self.assertIs(mds.interface, self.interface)
    with self.assertRaises(DataError):
        mds.iloc[3, 0]