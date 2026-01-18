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
def test_dataset_histogram_dask(self):
    import dask.array as da
    ds = Dataset((da.from_array(np.array(range(10), dtype='f'), chunks=3),), ['x'], datatype=['dask'])
    op_hist = histogram(ds, num_bins=3, normed=True)
    hist = Histogram(([0, 3, 6, 9], [0.1, 0.1, 0.133333]), vdims=('x_frequency', 'Frequency'))
    self.assertIsInstance(op_hist.data['x_frequency'], da.Array)
    self.assertEqual(op_hist, hist)