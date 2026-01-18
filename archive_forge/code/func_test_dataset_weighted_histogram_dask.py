import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
@da_skip
def test_dataset_weighted_histogram_dask(self):
    import dask.array as da
    ds = Dataset((da.from_array(np.array(range(10), dtype='f'), chunks=3), da.from_array([i / 10.0 for i in range(10)], chunks=3)), ['x', 'y'], datatype=['dask'])
    op_hist = histogram(ds, weight_dimension='y', num_bins=3, normed=True)
    hist = Histogram(([0, 3, 6, 9], [0.022222, 0.088889, 0.222222]), vdims='y')
    self.assertIsInstance(op_hist.data['y'], da.Array)
    self.assertEqual(op_hist, hist)