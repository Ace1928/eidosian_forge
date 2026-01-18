from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import PandasIndex
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
def rectify_dim_order(self, data, dataset) -> Dataset:
    return Dataset({k: v.transpose(*data[k].dims) for k, v in dataset.data_vars.items()}, dataset.coords, attrs=dataset.attrs)