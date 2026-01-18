from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_dataset_ellipsis_transpose_different_ordered_vars(self) -> None:
    ds = Dataset(dict(a=(('w', 'x', 'y', 'z'), np.ones((2, 3, 4, 5))), b=(('x', 'w', 'y', 'z'), np.zeros((3, 2, 4, 5)))))
    result = ds.transpose(..., 'z', 'y')
    assert list(result['a'].dims) == list('wxzy')
    assert list(result['b'].dims) == list('xwzy')