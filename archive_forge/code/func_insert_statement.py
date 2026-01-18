from __future__ import annotations
from abc import (
from contextlib import (
from datetime import (
from functools import partial
import re
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import get_option
from pandas.core.api import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime
def insert_statement(self, *, num_rows: int) -> str:
    names = list(map(str, self.frame.columns))
    wld = '?'
    escape = _get_valid_sqlite_name
    if self.index is not None:
        for idx in self.index[::-1]:
            names.insert(0, idx)
    bracketed_names = [escape(column) for column in names]
    col_names = ','.join(bracketed_names)
    row_wildcards = ','.join([wld] * len(names))
    wildcards = ','.join([f'({row_wildcards})' for _ in range(num_rows)])
    insert_statement = f'INSERT INTO {escape(self.name)} ({col_names}) VALUES {wildcards}'
    return insert_statement