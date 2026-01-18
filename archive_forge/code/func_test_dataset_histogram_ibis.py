import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
@ibis_skip
@pytest.mark.usefixtures('ibis_sqlite_backend')
def test_dataset_histogram_ibis(self):
    df = pd.DataFrame(dict(x=np.arange(10)))
    t = ibis.memtable(df, name='t')
    ds = Dataset(t, vdims='x')
    op_hist = histogram(ds, dimension='x', num_bins=3, normed=True)
    hist = Histogram(([0, 3, 6, 9], [0.1, 0.1, 0.133333]), vdims=('x_frequency', 'Frequency'))
    self.assertEqual(op_hist, hist)