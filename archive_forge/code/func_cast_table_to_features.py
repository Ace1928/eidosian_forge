import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
def cast_table_to_features(table: pa.Table, features: 'Features'):
    """Cast a table to the arrow schema that corresponds to the requested features.

    Args:
        table (`pyarrow.Table`):
            PyArrow table to cast.
        features ([`Features`]):
            Target features.

    Returns:
        table (`pyarrow.Table`): the casted table
    """
    if sorted(table.column_names) != sorted(features):
        raise CastError(f"Couldn't cast\n{table.schema}\nto\n{features}\nbecause column names don't match", table_column_names=table.column_names, requested_column_names=list(features))
    arrays = [cast_array_to_feature(table[name], feature) for name, feature in features.items()]
    return pa.Table.from_arrays(arrays, schema=features.arrow_schema)