from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
def generate_table(self) -> tuple[dict[str, tuple[int, int]], DataFrame]:
    """
        Generates the GSO lookup table for the DataFrame

        Returns
        -------
        gso_table : dict
            Ordered dictionary using the string found as keys
            and their lookup position (v,o) as values
        gso_df : DataFrame
            DataFrame where strl columns have been converted to
            (v,o) values

        Notes
        -----
        Modifies the DataFrame in-place.

        The DataFrame returned encodes the (v,o) values as uint64s. The
        encoding depends on the dta version, and can be expressed as

        enc = v + o * 2 ** (o_size * 8)

        so that v is stored in the lower bits and o is in the upper
        bits. o_size is

          * 117: 4
          * 118: 6
          * 119: 5
        """
    gso_table = self._gso_table
    gso_df = self.df
    columns = list(gso_df.columns)
    selected = gso_df[self.columns]
    col_index = [(col, columns.index(col)) for col in self.columns]
    keys = np.empty(selected.shape, dtype=np.uint64)
    for o, (idx, row) in enumerate(selected.iterrows()):
        for j, (col, v) in enumerate(col_index):
            val = row[col]
            val = '' if val is None else val
            key = gso_table.get(val, None)
            if key is None:
                key = (v + 1, o + 1)
                gso_table[val] = key
            keys[o, j] = self._convert_key(key)
    for i, col in enumerate(self.columns):
        gso_df[col] = keys[:, i]
    return (gso_table, gso_df)