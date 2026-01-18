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
@pytest.mark.parametrize('data, dtype', [('foo', np.dtype('U3')), (b'foo', np.dtype('S3'))])
def test_from_array_0d_string(self, data: Any, dtype: DTypeLike) -> None:
    named_array: NamedArray[Any, Any]
    named_array = from_array([], data)
    assert named_array.data == data
    assert named_array.dims == ()
    assert named_array.sizes == {}
    assert named_array.attrs == {}
    assert named_array.ndim == 0
    assert named_array.size == 1
    assert named_array.dtype == dtype