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
def test_cannot_change_mpl_aspect(self) -> None:
    with pytest.raises(ValueError, match='not available in xarray'):
        self.darray.plot.imshow(aspect='equal')
    self.darray.plot.imshow(size=5, aspect=2)
    assert 'auto' == plt.gca().get_aspect()
    assert tuple(plt.gcf().get_size_inches()) == (10, 5)