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
def schema_to_expression(schema: pa.Schema) -> pa.Schema:
    """Convert pyarrow.Schema to Triad schema expression
    see :func:`~triad.utils.pyarrow.expression_to_schema`

    :param schema: pyarrow schema
    :raises NotImplementedError: if there some type is not supported by Triad
    :return: schema string expression
    """
    return ','.join((_field_to_expression(x) for x in list(schema)))