import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_ndloc_slice_inverted_y(self):
    xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
    arr = np.arange(10) * np.arange(5)[np.newaxis].T
    ds = self.element((xs, ys[::-1], arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
    sliced = self.element((xs[2:5], ys[::-1][:-1], arr[:-1, 2:5]), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
    self.assertEqual(ds.ndloc[1:, 2:5], sliced)