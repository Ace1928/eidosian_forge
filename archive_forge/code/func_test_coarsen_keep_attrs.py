from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
def test_coarsen_keep_attrs(self, operation='mean'):
    _attrs = {'units': 'test', 'long_name': 'testing'}
    test_func = getattr(duck_array_ops, operation, None)
    with set_options(keep_attrs=False):
        new = Variable(['coord'], np.linspace(1, 10, 100), attrs=_attrs).coarsen(windows={'coord': 1}, func=test_func, boundary='exact', side='left')
    assert new.attrs == {}
    with set_options(keep_attrs=True):
        new = Variable(['coord'], np.linspace(1, 10, 100), attrs=_attrs).coarsen(windows={'coord': 1}, func=test_func, boundary='exact', side='left')
    assert new.attrs == _attrs