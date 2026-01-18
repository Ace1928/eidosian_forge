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
def test_broadcast_to_errors(self, target: NamedArray[Any, np.dtype[np.float32]]) -> None:
    with pytest.raises(ValueError, match='operands could not be broadcast together with remapped shapes'):
        target.broadcast_to({'x': 2, 'y': 2})
    with pytest.raises(ValueError, match='Cannot add new dimensions'):
        target.broadcast_to({'x': 2, 'y': 2, 'z': 2})