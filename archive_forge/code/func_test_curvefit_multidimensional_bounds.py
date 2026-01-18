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
def test_curvefit_multidimensional_bounds(self, use_dask: bool) -> None:
    if use_dask and (not has_dask):
        pytest.skip('requires dask')

    def sine(t, a, f, p):
        return a * np.sin(2 * np.pi * (f * t + p))
    t = np.arange(0, 2, 0.02)
    da = xr.DataArray(np.stack([sine(t, 1.0, 2, 0), sine(t, 1.0, 2, 0)]), coords={'x': [0, 1], 't': t})
    expected = DataArray([[1, 2, 0], [-1, 2, 0.5]], coords={'x': [0, 1], 'param': ['a', 'f', 'p']})
    if use_dask:
        da = da.chunk({'x': 1})
    fit = da.curvefit(coords=[da.t], func=sine, p0={'f': 2, 'p': 0.25}, bounds={'a': (DataArray([0, -2], coords=[da.x]), DataArray([2, 0], coords=[da.x]))})
    assert_allclose(fit.curvefit_coefficients, expected)
    fit2 = da.curvefit(coords=[da.t], func=sine, p0={'f': 2, 'p': 0.25}, bounds={'a': (-2, DataArray([2, 0], coords=[da.x]))})
    assert_allclose(fit2.curvefit_coefficients, expected)
    with pytest.raises(ValueError, match="Upper bound for 'a' has unexpected dimensions .* should only have dimensions that are in data dimensions"):
        da.curvefit(coords=[da.t], func=sine, bounds={'a': (0, DataArray([1], coords={'foo': [1]}))})