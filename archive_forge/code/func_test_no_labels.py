from __future__ import annotations
import contextlib
import inspect
import math
from collections.abc import Hashable
from copy import copy
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray.plot as xplt
from xarray import DataArray, Dataset
from xarray.namedarray.utils import module_available
from xarray.plot.dataarray_plot import _infer_interval_breaks
from xarray.plot.dataset_plot import _infer_meta_data
from xarray.plot.utils import (
from xarray.tests import (
def test_no_labels(self) -> None:
    self.darray.name = 'testvar'
    self.darray.attrs['units'] = 'test_units'
    self.plotmethod(add_labels=False)
    alltxt = text_in_fig()
    for string in ['x_long_name [x_units]', 'y_long_name [y_units]', 'testvar [test_units]']:
        assert string not in alltxt