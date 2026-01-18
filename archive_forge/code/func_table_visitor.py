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
def table_visitor(table: pa.Table, function: Callable[[pa.Array], None]):
    """Visit all arrays in a table and apply a function to them.

    Args:
        table (`pyarrow.Table`):
            PyArrow table to visit.
        function (`Callable[[pa.Array], None]`):
            Function to apply to each array.
    """
    from .features import Features, Sequence
    features = Features.from_arrow_schema(table.schema)

    def _visit(array, feature):
        if isinstance(array, pa.ChunkedArray):
            for chunk in array.chunks:
                _visit(chunk, feature)
        else:
            if isinstance(array, pa.ExtensionArray):
                array = array.storage
            function(array, feature)
            if pa.types.is_struct(array.type) and (not hasattr(feature, 'cast_storage')):
                if isinstance(feature, Sequence) and isinstance(feature.feature, dict):
                    feature = {name: Sequence(subfeature, length=feature.length) for name, subfeature in feature.feature.items()}
                for name, subfeature in feature.items():
                    _visit(array.field(name), subfeature)
            elif pa.types.is_list(array.type):
                if isinstance(feature, list):
                    _visit(array.values, feature[0])
                elif isinstance(feature, Sequence):
                    _visit(array.values, feature.feature)
    for name, feature in features.items():
        _visit(table[name], feature)