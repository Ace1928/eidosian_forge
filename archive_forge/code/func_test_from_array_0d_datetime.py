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
def test_from_array_0d_datetime(self) -> None:
    named_array: NamedArray[Any, Any]
    named_array = from_array([], np.datetime64('2000-01-01'))
    assert named_array.dtype == np.dtype('datetime64[D]')