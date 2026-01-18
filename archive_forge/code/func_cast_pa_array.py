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
def cast_pa_array(col: pa.Array, new_type: pa.DataType) -> pa.Array:
    old_type = col.type
    if new_type.equals(old_type):
        return col
    elif pa.types.is_date(new_type) and (not pa.types.is_date(old_type)):
        return pa.Array.from_pandas(pd.to_datetime(col.to_pandas()).dt.date)
    elif pa.types.is_timestamp(new_type):
        if pa.types.is_timestamp(old_type) or pa.types.is_date(old_type):
            s = pd.to_datetime(col.to_pandas())
            from_tz = old_type.tz if pa.types.is_timestamp(old_type) else None
            to_tz = new_type.tz
            if from_tz is None or to_tz is None:
                s = s.dt.tz_localize(to_tz)
            else:
                s = s.dt.tz_convert(to_tz)
        else:
            s = pd.to_datetime(col.to_pandas())
        return pa.Array.from_pandas(s, type=new_type)
    elif pa.types.is_integer(new_type):
        if PYARROW_VERSION.major < 9:
            return col.cast(new_type, safe=False)
        return col.cast(options=CastOptions(new_type, allow_decimal_truncate=True, allow_float_truncate=True))
    elif pa.types.is_string(new_type):
        if pa.types.is_timestamp(old_type):
            series = pd.to_datetime(col.to_pandas())
            ns = series.isnull()
            series = series.astype(str)
            return pa.Array.from_pandas(series.mask(ns, None), type=new_type)
    return col.cast(new_type, safe=True)