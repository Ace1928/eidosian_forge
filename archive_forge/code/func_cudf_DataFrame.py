from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def cudf_DataFrame(*args, **kwargs):
    assert not kwargs.pop('geo', False)
    return cudf.DataFrame.from_pandas(pd.DataFrame(*args, **kwargs), nan_as_null=False)