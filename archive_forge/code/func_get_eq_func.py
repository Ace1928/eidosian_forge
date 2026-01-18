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
def get_eq_func(data_type: pa.DataType) -> Callable[[Any, Any], bool]:
    """Generate equality function for a give datatype

    :param data_type: pyarrow data type supported by Triad
    :return: the function
    """
    is_supported(data_type, throw=True)
    if data_type in _COMPARATORS:
        return _COMPARATORS[data_type]
    if pa.types.is_date(data_type):
        return _date_eq
    if pa.types.is_timestamp(data_type):
        return _timestamp_eq
    return _general_eq