from unittest import TestCase, SkipTest
import numpy as np
from hvplot.plotting import hvPlotTabular, hvPlot
def test_dask_series_patched(self):
    import pandas as pd
    import dask.dataframe as dd
    series = pd.Series([0, 1, 2])
    dseries = dd.from_pandas(series, 2)
    self.assertIsInstance(dseries.hvplot, hvPlotTabular)