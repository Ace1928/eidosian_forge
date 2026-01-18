import sqlite3
from tempfile import NamedTemporaryFile
from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.ibis import IbisInterface
from holoviews.core.spaces import HoloMap
from .base import HeterogeneousColumnTests, InterfaceTests, ScalarColumnTests
def test_aggregation_operations(self):
    for agg in [np.min, np.nanmin, np.max, np.nanmax, np.mean, np.nanmean, np.sum, np.nansum, len, np.count_nonzero]:
        data = self.table.dframe()
        expected = self.table.clone(data=data).aggregate('Gender', agg).sort()
        result = self.table.aggregate('Gender', agg).sort()
        self.compare_dataset(expected, result, msg=str(agg))