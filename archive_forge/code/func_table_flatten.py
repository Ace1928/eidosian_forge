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
def table_flatten(table: pa.Table):
    """Improved version of `pa.Table.flatten`.

    It behaves as `pa.Table.flatten` in a sense it does 1-step flatten of the columns with a struct type into one column per struct field,
    but updates the metadata and skips decodable features unless the `decode` attribute of these features is set to False.

    Args:
        table (`pa.Table`):
            PyArrow table to flatten.

    Returns:
        `Table`: the flattened table
    """
    from .features import Features
    features = Features.from_arrow_schema(table.schema)
    if any((hasattr(subfeature, 'flatten') and subfeature.flatten() == subfeature for subfeature in features.values())):
        flat_arrays = []
        flat_column_names = []
        for field in table.schema:
            array = table.column(field.name)
            subfeature = features[field.name]
            if pa.types.is_struct(field.type) and (not hasattr(subfeature, 'flatten') or subfeature.flatten() != subfeature):
                flat_arrays.extend(array.flatten())
                flat_column_names.extend([f'{field.name}.{subfield.name}' for subfield in field.type])
            else:
                flat_arrays.append(array)
                flat_column_names.append(field.name)
        flat_table = pa.Table.from_arrays(flat_arrays, names=flat_column_names)
    else:
        flat_table = table.flatten()
    flat_features = features.flatten(max_depth=2)
    flat_features = Features({column_name: flat_features[column_name] for column_name in flat_table.column_names})
    return flat_table.replace_schema_metadata(flat_features.arrow_schema.metadata)