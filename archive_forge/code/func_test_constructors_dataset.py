from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_constructors_dataset(self):
    ds = Dataset(self.df)
    self.assertIs(ds, ds.dataset)
    ops = ds.pipeline.operations
    self.assertEqual(len(ops), 1)
    self.assertIs(ops[0].output_type, Dataset)
    self.assertEqual(ds, ds.pipeline(ds.dataset))