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
def test_facetgrid(self) -> None:
    with figure_context():
        fg = self.ds.plot.streamplot(x='x', y='y', u='u', v='v', row='row', col='col', hue='mag')
        for handle in fg._mappables:
            assert isinstance(handle, mpl.collections.LineCollection)
    with figure_context():
        fg = self.ds.plot.streamplot(x='x', y='y', u='u', v='v', row='row', col='col', hue='mag', add_guide=False)