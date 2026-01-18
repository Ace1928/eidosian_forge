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
def pa_datatypes_equal(t1: pa.DataType, t2: pa.DataType, ignore_list_item_name: bool=True, equal_groups: Optional[List[List[Callable[[pa.DataType], bool]]]]=None) -> bool:
    """Check if two pyarrow data types are equal

    :param t1: the first pyarrow data type
    :param t2: the second pyarrow data type
    :param ignore_list_item_name: whether to ignore list item name,
        defaults to True
    :param equal_groups: a list of groups of functions to check equality,
        defaults to None

    :return: if the two data types are equal

    .. note::

        In the lastest version of pyarrow, in the default comparison logic,
        list field names are not compared.

    .. admonition:: Examples

        .. code-block:: python

            assert not pa_datatypes_equal(pa.int32(), pa.int64())
            assert pa_datatypes_equal(
                pa.int32(),
                pa.int64(),
                equal_groups=[[pa.types.is_integer]],
            )
    """
    if t1 is t2:
        return True
    if not ignore_list_item_name and pa.types.is_list(t1) and pa.types.is_list(t2) and (t1.value_field.name != t2.value_field.name):
        return False
    if t1 == t2:
        return True
    if equal_groups is not None:
        for group in equal_groups:
            if any((f(t1) for f in group)) and any((f(t2) for f in group)):
                return True
    params: Dict[str, Any] = dict(ignore_list_item_name=ignore_list_item_name, equal_groups=equal_groups)
    if pa.types.is_list(t1) and pa.types.is_list(t2):
        return pa_datatypes_equal(t1.value_type, t2.value_type, **params)
    if pa.types.is_struct(t1) and pa.types.is_struct(t2):
        if len(t1) != len(t2):
            return False
        for f1, f2 in zip(t1, t2):
            if f1.name != f2.name:
                return False
            if not pa_datatypes_equal(f1.type, f2.type, **params):
                return False
        return True
    if pa.types.is_map(t1) and pa.types.is_map(t2):
        return pa_datatypes_equal(t1.key_type, t2.key_type, **params) and pa_datatypes_equal(t1.item_type, t2.item_type, **params)
    return False