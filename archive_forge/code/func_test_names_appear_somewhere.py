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
@pytest.mark.slow
def test_names_appear_somewhere(self) -> None:
    self.darray.name = 'testvar'
    self.g.map_dataarray(xplt.contourf, 'x', 'y')
    for k, ax in zip('abc', self.g.axs.flat):
        assert f'z = {k}' == ax.get_title()
    alltxt = text_in_fig()
    assert self.darray.name in alltxt
    for label in ['x', 'y']:
        assert label in alltxt