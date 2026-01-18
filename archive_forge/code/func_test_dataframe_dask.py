from packaging.version import Version
from unittest import SkipTest
import numpy as np
import pandas as pd
import hvplot.pandas  # noqa
import pytest
from hvplot import hvPlotTabular
from hvplot.tests.util import makeDataFrame
@pytest.mark.skipif(dd is None, reason='dask not installed')
@frame_kinds
@y_combinations
def test_dataframe_dask(kind, y):
    df = dd.from_pandas(makeDataFrame(), npartitions=2)
    assert isinstance(df, dd.DataFrame)
    df.hvplot(y=y, kind=kind)