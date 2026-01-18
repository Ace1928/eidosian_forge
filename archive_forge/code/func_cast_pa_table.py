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
def cast_pa_table(df: pa.Table, schema: pa.Schema) -> pa.Table:
    """Convert a pyarrow table to another pyarrow table with given schema

    :param df: the pyarrow table
    :param schema: the pyarrow schema
    :return: the converted pyarrow table
    """
    if df.schema == schema:
        return df
    cols = [cast_pa_array(col, new_f.type) for col, new_f in zip(df.columns, schema)]
    return pa.Table.from_arrays(cols, schema=schema)