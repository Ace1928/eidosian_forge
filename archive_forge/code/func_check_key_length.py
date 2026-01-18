from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
def check_key_length(columns: Index, key, value: DataFrame) -> None:
    """
    Checks if a key used as indexer has the same length as the columns it is
    associated with.

    Parameters
    ----------
    columns : Index The columns of the DataFrame to index.
    key : A list-like of keys to index with.
    value : DataFrame The value to set for the keys.

    Raises
    ------
    ValueError: If the length of key is not equal to the number of columns in value
                or if the number of columns referenced by key is not equal to number
                of columns.
    """
    if columns.is_unique:
        if len(value.columns) != len(key):
            raise ValueError('Columns must be same length as key')
    elif len(columns.get_indexer_non_unique(key)[0]) != len(value.columns):
        raise ValueError('Columns must be same length as key')