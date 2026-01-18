import json
import pickle
from datetime import date, datetime
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import io
import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version
from pandas.core.dtypes.base import ExtensionDtype
from pyarrow.compute import CastOptions, binary_join_element_wise
from pyarrow.json import read_json, ParseOptions as JsonParseOptions
from triad.constants import TRIAD_VAR_QUOTE
from .convert import as_type
from .iter import EmptyAwareIterable, Slicer
from .json import loads_no_dup
from .schema import move_to_unquoted, quote_name, unquote_name
from .assertion import assert_or_throw
def replace_types_in_table(df: pa.Table, pairs: List[Tuple[Union[Callable[[pa.DataType], bool], pa.DataType], Union[Callable[[pa.DataType], pa.DataType], pa.DataType]]], recursive: bool=True, safe: bool=True) -> pa.Table:
    """Replace(cast) types in a table

    :param df: the table
    :param pairs: a list of (is_type, convert_type) pairs
    :param recursive: whether to do recursive replacement in nested types
    :param safe: whether to check for conversion errors such as overflow

    :return: the new table
    """
    old_schema = df.schema
    new_schema = replace_types_in_schema(old_schema, pairs, recursive)
    if old_schema is new_schema:
        return df
    func = get_alter_func(old_schema, new_schema, safe=safe)
    return func(df)