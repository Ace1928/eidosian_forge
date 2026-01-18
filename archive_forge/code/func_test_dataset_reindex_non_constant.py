import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
def test_dataset_reindex_non_constant(self):
    with DatatypeContext([self.datatype, 'dictionary', 'dataframe', 'grid'], self.rgb):
        ds = Dataset(self.rgb)
        reindexed = ds.reindex(['y'], ['R'])
    data = Dataset(ds.columns(['y', 'R']), kdims=['y'], vdims=[ds.vdims[0]])
    self.assertEqual(reindexed, data)