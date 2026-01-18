from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import deepcopy
from textwrap import dedent
from typing import Any, Final, Literal, cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import (
from xarray.coding.times import CFDatetimeCoder
from xarray.core import dtypes
from xarray.core.common import full_like
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import Index, PandasIndex, filter_indexes_from_coords
from xarray.core.types import QueryEngineOptions, QueryParserOptions
from xarray.core.utils import is_scalar
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
@requires_scipy
@pytest.mark.parametrize('use_dask', [True, False])
def test_curvefit_ignore_errors(self, use_dask: bool) -> None:
    if use_dask and (not has_dask):
        pytest.skip('requires dask')

    def line(x, a, b):
        if a > 10:
            return 0
        return a * x + b
    da = DataArray([[1, 3, 5], [0, 20, 40]], coords={'i': [1, 2], 'x': [0.0, 1.0, 2.0]})
    if use_dask:
        da = da.chunk({'i': 1})
    expected = DataArray([[2, 1], [np.nan, np.nan]], coords={'i': [1, 2], 'param': ['a', 'b']})
    with pytest.raises(RuntimeError, match='calls to function has reached maxfev'):
        da.curvefit(coords='x', func=line, kwargs=dict(maxfev=5)).compute()
    fit = da.curvefit(coords='x', func=line, errors='ignore', kwargs=dict(maxfev=5)).compute()
    assert_allclose(fit.curvefit_coefficients, expected)