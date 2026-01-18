from __future__ import annotations
import copy
import warnings
from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, cast, overload
import numpy as np
import pytest
from xarray.core.indexing import ExplicitlyIndexed
from xarray.namedarray._typing import (
from xarray.namedarray.core import NamedArray, from_array
def test_from_array_0d_object(self) -> None:
    named_array: NamedArray[Any, Any]
    named_array = from_array([], (10, 12, 12))
    expected_data = np.empty((), dtype=object)
    expected_data[()] = (10, 12, 12)
    assert np.array_equal(np.asarray(named_array.data), expected_data)
    assert named_array.dims == ()
    assert named_array.sizes == {}
    assert named_array.attrs == {}
    assert named_array.ndim == 0
    assert named_array.size == 1
    assert named_array.dtype == np.dtype('O')