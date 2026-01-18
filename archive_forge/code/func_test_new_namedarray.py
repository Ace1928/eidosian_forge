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
def test_new_namedarray(self) -> None:
    dtype_float = np.dtype(np.float32)
    narr_float: NamedArray[Any, np.dtype[np.float32]]
    narr_float = NamedArray(('x',), np.array([1.5, 3.2], dtype=dtype_float))
    assert narr_float.dtype == dtype_float
    dtype_int = np.dtype(np.int8)
    narr_int: NamedArray[Any, np.dtype[np.int8]]
    narr_int = narr_float._new(('x',), np.array([1, 3], dtype=dtype_int))
    assert narr_int.dtype == dtype_int

    class Variable(NamedArray[_ShapeType_co, _DType_co], Generic[_ShapeType_co, _DType_co]):

        @overload
        def _new(self, dims: _DimsLike | Default=..., data: duckarray[Any, _DType]=..., attrs: _AttrsLike | Default=...) -> Variable[Any, _DType]:
            ...

        @overload
        def _new(self, dims: _DimsLike | Default=..., data: Default=..., attrs: _AttrsLike | Default=...) -> Variable[_ShapeType_co, _DType_co]:
            ...

        def _new(self, dims: _DimsLike | Default=_default, data: duckarray[Any, _DType] | Default=_default, attrs: _AttrsLike | Default=_default) -> Variable[Any, _DType] | Variable[_ShapeType_co, _DType_co]:
            dims_ = copy.copy(self._dims) if dims is _default else dims
            attrs_: Mapping[Any, Any] | None
            if attrs is _default:
                attrs_ = None if self._attrs is None else self._attrs.copy()
            else:
                attrs_ = attrs
            if data is _default:
                return type(self)(dims_, copy.copy(self._data), attrs_)
            cls_ = cast('type[Variable[Any, _DType]]', type(self))
            return cls_(dims_, data, attrs_)
    var_float: Variable[Any, np.dtype[np.float32]]
    var_float = Variable(('x',), np.array([1.5, 3.2], dtype=dtype_float))
    assert var_float.dtype == dtype_float
    var_int: Variable[Any, np.dtype[np.int8]]
    var_int = var_float._new(('x',), np.array([1, 3], dtype=dtype_int))
    assert var_int.dtype == dtype_int