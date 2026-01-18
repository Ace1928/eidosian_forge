from collections.abc import Mapping, MutableMapping
from functools import partial
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version
from .. import config
from ..features import Features
from ..features.features import _ArrayXDExtensionType, _is_zero_copy_only, decode_nested_example, pandas_types_mapper
from ..table import Table
from ..utils.py_utils import no_op_if_value_is_null
def format_column(self, pa_table: pa.Table) -> ColumnFormat:
    formatted_batch = self.format_batch(pa_table)
    if hasattr(formatted_batch, 'keys'):
        if len(formatted_batch.keys()) > 1:
            raise TypeError(f'Tried to query a column but the custom formatting function returns too many columns. Only one column was expected but got columns {list(formatted_batch.keys())}.')
    else:
        raise TypeError(f'Custom formatting function must return a dict to be able to pick a row, but got {formatted_batch}')
    try:
        return formatted_batch[pa_table.column_names[0]]
    except Exception as exc:
        raise TypeError(f'Custom formatting function must return a dict to be able to pick a row, but got {formatted_batch}') from exc