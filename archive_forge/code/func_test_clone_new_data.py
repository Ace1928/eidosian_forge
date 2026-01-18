from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_clone_new_data(self):
    ds_clone = self.ds.clone(data=self.ds2.data)
    self.assertEqual(ds_clone.dataset, self.ds2)
    self.assertEqual(len(ds_clone.pipeline.operations), 1)