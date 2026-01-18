import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_dataset_histogram_empty_explicit_bins(self):
    ds = Dataset([np.nan, np.nan], ['x'])
    op_hist = histogram(ds, bins=[0, 1, 2])
    hist = Histogram(([0, 1, 2], [0, 0]), vdims=('x_count', 'Count'))
    self.assertEqual(op_hist, hist)