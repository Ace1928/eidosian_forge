from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
def mean_func(ds):
    return ds.weighted(ds.weights).mean('time')