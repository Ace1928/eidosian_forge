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
def test_from_array_with_0d_object(self) -> None:
    data = np.empty((), dtype=object)
    data[()] = (10, 12, 12)
    narr = from_array((), data)
    np.array_equal(np.asarray(narr.data), data)