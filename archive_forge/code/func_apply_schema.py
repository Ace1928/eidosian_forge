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
def apply_schema(schema: pa.Schema, data: Iterable[List[Any]], copy: bool=True, deep: bool=False, str_as_json: bool=True) -> Iterable[List[Any]]:
    """Use `pa.Schema` to convert a row(list) to the correspondent types.

    Notice this function is to convert from python native type to python
    native type. It is used to normalize data input, which could be generated
    by different logics, into the correct data types.

    Notice this function assumes each item of `data` has the same length
    with `schema` and will not do any extra validation on that.

    :param schema: pyarrow schema
    :param data: and iterable of rows, represtented by list or tuple
    :param copy: whether to apply inplace (copy=False), or create new instances
    :param deep: whether to do deep conversion on nested (struct, list) types
    :param str_as_json: where to treat string data as json for nested types

    :raises ValueError: if any value can't be converted to the datatype
    :raises NotImplementedError: if any field type is not supported by Triad

    :yield: converted rows
    """
    converter = _TypeConverter(schema, copy=copy, deep=deep, str_as_json=str_as_json)
    try:
        for item in data:
            yield converter.row_to_py(item)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(str(e))