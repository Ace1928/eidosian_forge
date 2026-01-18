import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_nodata_range(self):
    ds = self.dataset_grid.clone(vdims=[Dimension('z', nodata=0)])
    self.assertEqual(ds.range('z'), (1, 5))