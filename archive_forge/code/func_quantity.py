from __future__ import annotations
import warnings
import numpy as np
import pytest
import xarray as xr
from xarray.tests import has_dask
def quantity(x):
    return x