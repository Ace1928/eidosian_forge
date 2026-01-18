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
def test_equals_all_dtypes(self):
    for v, _ in self.example_1d_objects():
        v2 = v.copy()
        assert v.equals(v2)
        assert v.identical(v2)
        assert v.no_conflicts(v2)
        assert v[0].equals(v2[0])
        assert v[0].identical(v2[0])
        assert v[0].no_conflicts(v2[0])
        assert v[:2].equals(v2[:2])
        assert v[:2].identical(v2[:2])
        assert v[:2].no_conflicts(v2[:2])