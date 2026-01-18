from __future__ import annotations
import itertools
import textwrap
import warnings
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime
from inspect import getfullargspec
from typing import TYPE_CHECKING, Any, Callable, Literal, overload
import numpy as np
import pandas as pd
from xarray.core.indexes import PandasMultiIndex
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar, module_available
from xarray.namedarray.pycompat import DuckArrayModule
@property
def data_is_numeric(self) -> bool:
    """
        Check if data is numeric.

        Examples
        --------
        >>> a = xr.DataArray(["b", "a", "a", "b", "c"])
        >>> _Normalize(a).data_is_numeric
        False

        >>> a = xr.DataArray([0.5, 0, 0, 0.5, 2, 3])
        >>> _Normalize(a).data_is_numeric
        True

        >>> # TODO: Datetime should be numeric right?
        >>> a = xr.DataArray(pd.date_range("2000-1-1", periods=4))
        >>> _Normalize(a).data_is_numeric
        False

        # TODO: Timedelta should be numeric right?
        >>> a = xr.DataArray(pd.timedelta_range("-1D", periods=4, freq="D"))
        >>> _Normalize(a).data_is_numeric
        True
        """
    return self._data_is_numeric