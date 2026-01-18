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
def test_duck_array_class(self) -> None:

    def test_duck_array_typevar(a: duckarray[Any, _DType]) -> duckarray[Any, _DType]:
        b: duckarray[Any, _DType] = a
        if isinstance(b, _arrayfunction_or_api):
            return b
        else:
            raise TypeError(f'a ({type(a)}) is not a valid _arrayfunction or _arrayapi')
    numpy_a: NDArray[np.int64]
    numpy_a = np.array([2.1, 4], dtype=np.dtype(np.int64))
    test_duck_array_typevar(numpy_a)
    masked_a: np.ma.MaskedArray[Any, np.dtype[np.int64]]
    masked_a = np.ma.asarray([2.1, 4], dtype=np.dtype(np.int64))
    test_duck_array_typevar(masked_a)
    custom_a: CustomArrayIndexable[Any, np.dtype[np.int64]]
    custom_a = CustomArrayIndexable(numpy_a)
    test_duck_array_typevar(custom_a)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'The numpy.array_api submodule is still experimental', category=UserWarning)
        import numpy.array_api as nxp
    arrayapi_a: duckarray[Any, Any]
    arrayapi_a = nxp.asarray([2.1, 4], dtype=np.dtype(np.int64))
    test_duck_array_typevar(arrayapi_a)