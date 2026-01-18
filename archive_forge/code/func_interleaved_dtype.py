from __future__ import annotations
from typing import (
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas.errors import AbstractMethodError
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.dtypes import (
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.indexes.api import (
def interleaved_dtype(dtypes: list[DtypeObj]) -> DtypeObj | None:
    """
    Find the common dtype for `blocks`.

    Parameters
    ----------
    blocks : List[DtypeObj]

    Returns
    -------
    dtype : np.dtype, ExtensionDtype, or None
        None is returned when `blocks` is empty.
    """
    if not len(dtypes):
        return None
    return find_common_type(dtypes)