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
@requires_iris
@pytest.mark.parametrize('var_name, std_name, long_name, name, attrs', [('var_name', 'height', 'Height', 'var_name', {'standard_name': 'height', 'long_name': 'Height'}), (None, 'height', 'Height', 'height', {'standard_name': 'height', 'long_name': 'Height'}), (None, None, 'Height', 'Height', {'long_name': 'Height'}), (None, None, None, 'unknown', {})])
def test_da_coord_name_from_cube(self, std_name, long_name, var_name, name, attrs) -> None:
    from iris.coords import DimCoord
    from iris.cube import Cube
    latitude = DimCoord([-90, 0, 90], standard_name=std_name, var_name=var_name, long_name=long_name)
    data = [0, 0, 0]
    cube = Cube(data, dim_coords_and_dims=[(latitude, 0)])
    result = xr.DataArray.from_iris(cube)
    expected = xr.DataArray(data, coords=[(name, [-90, 0, 90], attrs)])
    xr.testing.assert_identical(result, expected)