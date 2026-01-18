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
def schemas_equal(a: pa.Schema, b: pa.Schema, check_order: bool=True, check_metadata: bool=True, ignore: Optional[List[Tuple[Union[Callable[[pa.DataType], bool], pa.DataType], Union[Callable[[pa.DataType], pa.DataType], pa.DataType]]]]=None) -> bool:
    """check if two schemas are equal

    :param a: first pyarrow schema
    :param b: second pyarrow schema
    :param compare_order: whether to compare order
    :param compare_order: whether to compare metadata
    :param ignore: a list of (is_type, convert_type) pairs to
        ignore differences on, defaults to None
    :return: if the two schema equal
    """
    if a is b:
        return True
    if ignore is not None:
        a = replace_types_in_schema(a, ignore, recursive=True)
        b = replace_types_in_schema(b, ignore, recursive=True)
    if check_order:
        return a.equals(b, check_metadata=check_metadata)
    if check_metadata and a.metadata != b.metadata:
        return False
    da = {k: a.field(k) for k in a.names}
    db = {k: b.field(k) for k in b.names}
    return da == db