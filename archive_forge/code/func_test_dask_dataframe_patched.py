from unittest import TestCase, SkipTest
import numpy as np
from hvplot.plotting import hvPlotTabular, hvPlot
def test_dask_dataframe_patched(self):
    import pandas as pd
    import dask.dataframe as dd
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['x', 'y'])
    ddf = dd.from_pandas(df, 2)
    self.assertIsInstance(ddf.hvplot, hvPlotTabular)