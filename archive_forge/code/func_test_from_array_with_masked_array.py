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
def test_from_array_with_masked_array(self) -> None:
    masked_array: np.ndarray[Any, np.dtype[np.generic]]
    masked_array = np.ma.array([1, 2, 3], mask=[False, True, False])
    with pytest.raises(NotImplementedError):
        from_array(('x',), masked_array)