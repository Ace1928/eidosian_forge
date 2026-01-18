from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_gpu_dependencies():
    if test_gpu and cudf is None:
        pytest.fail('cudf and/or cupy not available and DATASHADER_TEST_GPU=1')