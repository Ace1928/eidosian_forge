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
def format_row(self, pa_table: pa.Table) -> dict:
    formatted_batch = self.format_batch(pa_table)
    try:
        return _unnest(formatted_batch)
    except Exception as exc:
        raise TypeError(f'Custom formatting function must return a dict of sequences to be able to pick a row, but got {formatted_batch}') from exc