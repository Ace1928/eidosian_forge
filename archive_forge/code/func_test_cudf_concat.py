from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.skipif(not test_gpu, reason='DATASHADER_TEST_GPU not set')
def test_cudf_concat():
    with pytest.raises(NotImplementedError):
        dfp = pd.DataFrame({'y': [0, 1]})
        dfc = cudf.from_pandas(dfp)
        cudf.concat((dfc['y'], dfc['y']), axis=1)