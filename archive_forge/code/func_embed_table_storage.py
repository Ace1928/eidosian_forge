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
def embed_table_storage(table: pa.Table):
    """Embed external data into a table's storage.

    <Added version="2.4.0"/>

    Args:
        table (`pyarrow.Table`):
            PyArrow table in which to embed data.

    Returns:
        table (`pyarrow.Table`): the table with embedded data
    """
    from .features.features import Features, require_storage_embed
    features = Features.from_arrow_schema(table.schema)
    arrays = [embed_array_storage(table[name], feature) if require_storage_embed(feature) else table[name] for name, feature in features.items()]
    return pa.Table.from_arrays(arrays, schema=features.arrow_schema)