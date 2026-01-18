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
def parse_json_columns(df: pa.Table, columns: Union[List[str], pa.Schema]) -> pa.Table:
    """Parse json string columns in a table and replace them with
    pyarrow types.

    :param df: the table
    :param columns: the columns to convert, can be a list of column names
        or a schema. If it is a list of names, then their types will be
        inferred from the data.
    :return: the new table
    """
    if isinstance(columns, list):
        sub = df.select(columns)
        options: Any = None
    else:
        sub = df.select([f.name for f in columns])
        schema = pa.schema([pa.field(f'col{i}', f.type) for i, f in enumerate(columns)])
        options = JsonParseOptions(explicit_schema=schema)
    if sub.num_columns == 0:
        return df
    assert_or_throw(all((pa.types.is_string(tp) for tp in sub.schema.types)), ValueError('all selected columns must be string'))
    name_map: Dict[str, int] = {}
    args: List[Any] = []
    for i, name in enumerate(sub.column_names):
        key = f'"col{i}":'
        if i > 0:
            key = ',' + key
        else:
            key = '{' + key
        args.append(key)
        args.append(sub.column(i))
        name_map[name] = i
        if i == len(sub.column_names) - 1:
            args.append('}')
    jcol = binary_join_element_wise(args[0], args[1], '', null_handling='replace', null_replacement='null')
    for i in range(2, len(args)):
        jcol = binary_join_element_wise(jcol, args[i], '', null_handling='replace', null_replacement='null')
    json_stream = io.BytesIO(str.encode('\n'.join(jcol.to_pylist())))
    parsed = read_json(json_stream, parse_options=options)
    cols: List[Any] = []
    for name, col in zip(df.schema.names, df.columns):
        if name not in name_map:
            cols.append(col)
        else:
            cols.append(parsed.column(name_map[name]))
    return pa.Table.from_arrays(cols, names=df.column_names)