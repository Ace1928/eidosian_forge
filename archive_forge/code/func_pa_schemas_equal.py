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
def pa_schemas_equal(s1: pa.Schema, s2: pa.Schema, ignore_list_item_name: bool=True, equal_groups: Optional[List[List[Callable[[pa.DataType], bool]]]]=None) -> bool:
    """Check if two pyarrow schemas are equal

    :param s1: the first pyarrow schema
    :param s2: the second pyarrow schema
    :param ignore_list_item_name: whether to ignore list item name,
        defaults to True
    :param equal_groups: a list of groups of functions to check equality,
        defaults to None

    :return: if the two schemas are equal

    .. note::

        In the lastest version of pyarrow, in the default comparison logic,
        list field names are not compared.

    .. admonition:: Examples

        .. code-block:: python

            s1 = pa.schema([("a", pa.int32()), ("b", pa.string())])
            s2 = pa.schema([("a", pa.int64()), ("b", pa.string())])
            assert not pa_schemas_equal(s1, s2)
            assert pa_schemas_equal(
                s1,
                s2,
                equal_groups=[[pa.types.is_integer]],
            )
    """
    if ignore_list_item_name:
        if s1 is s2 or s1.equals(s2):
            return True
    elif s1 is s2:
        return True
    if s1.names != s2.names:
        return False
    for f1, f2 in zip(s1, s2):
        if not pa_datatypes_equal(f1.type, f2.type, ignore_list_item_name=ignore_list_item_name, equal_groups=equal_groups):
            return False
    return True