from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import MultiBaseInterfaceTest
from geoviews.data.geom_dict import GeomDictInterface
def test_dict_dataset(self):
    dicts = [{'x': np.arange(i, i + 2), 'y': np.arange(i, i + 2)} for i in range(2)]
    mds = Path(dicts, kdims=['x', 'y'], datatype=[self.datatype])
    self.assertIs(mds.interface, self.interface)
    for i, cols in enumerate(mds.split(datatype='columns')):
        self.assertEqual(dict(cols), dict(dicts[i], geom_type='Line', geometry=mds.data[i]['geometry']))