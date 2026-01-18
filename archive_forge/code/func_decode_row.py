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
def decode_row(self, row: pd.DataFrame) -> pd.DataFrame:
    decode = {column_name: no_op_if_value_is_null(partial(decode_nested_example, feature)) for column_name, feature in self.features.items() if self.features._column_requires_decoding[column_name]} if self.features else {}
    if decode:
        row[list(decode.keys())] = row.transform(decode)
    return row