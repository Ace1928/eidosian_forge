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
@pytest.mark.parametrize('dims, data_shape, new_dims, raises', [(['x', 'y', 'z'], (2, 3, 4), ['a', 'b', 'c'], False), (['x', 'y', 'z'], (2, 3, 4), ['a', 'b'], True), (['x', 'y', 'z'], (2, 4, 5), ['a', 'b', 'c', 'd'], True), ([], [], (), False), ([], [], ('x',), True)])
def test_dims_setter(self, dims: Any, data_shape: Any, new_dims: Any, raises: bool) -> None:
    named_array: NamedArray[Any, Any]
    named_array = NamedArray(dims, np.asarray(np.random.random(data_shape)))
    assert named_array.dims == tuple(dims)
    if raises:
        with pytest.raises(ValueError):
            named_array.dims = new_dims
    else:
        named_array.dims = new_dims
        assert named_array.dims == tuple(new_dims)