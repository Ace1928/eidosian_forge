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
def test_copy_index_with_data_errors(self) -> None:
    orig = IndexVariable('x', np.arange(5))
    new_data = np.arange(5, 20)
    with pytest.raises(ValueError, match='must match shape of object'):
        orig.copy(data=new_data)
    with pytest.raises(ValueError, match='Cannot assign to the .data'):
        orig.data = new_data
    with pytest.raises(ValueError, match='Cannot assign to the .values'):
        orig.values = new_data