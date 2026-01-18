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
def test_facetgrid_hue_style(self) -> None:
    ds2 = self.ds.copy()
    g = ds2.plot.scatter(x='A', y='B', row='row', col='col', hue='hue')
    assert isinstance(g._mappables[-1], mpl.collections.PathCollection)
    ds2['hue'] = pd.date_range('2000-1-1', periods=4)
    g = ds2.plot.scatter(x='A', y='B', row='row', col='col', hue='hue')
    assert isinstance(g._mappables[-1], mpl.collections.PathCollection)
    ds2['hue'] = ['a', 'a', 'b', 'b']
    g = ds2.plot.scatter(x='A', y='B', row='row', col='col', hue='hue')
    assert isinstance(g._mappables[-1], mpl.collections.PathCollection)