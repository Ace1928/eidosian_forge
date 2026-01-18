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
@_wrap_for_chunked_arrays
def embed_array_storage(array: pa.Array, feature: 'FeatureType'):
    """Embed data into an arrays's storage.
    For custom features like Audio or Image, it takes into account the "embed_storage" methods
    they define to embed external data (e.g. an image file) into an array.

    <Added version="2.4.0"/>

    Args:
        array (`pa.Array`):
            The PyArrow array in which to embed data.
        feature (`datasets.features.FeatureType`):
            Array features.

    Raises:
        `TypeError`: if the target type is not supported according, e.g.

            - if a field is missing

    Returns:
         array (`pyarrow.Array`): the casted array
    """
    from .features import Sequence
    _e = embed_array_storage
    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if hasattr(feature, 'embed_storage'):
        return feature.embed_storage(array)
    elif pa.types.is_struct(array.type):
        if isinstance(feature, Sequence) and isinstance(feature.feature, dict):
            feature = {name: Sequence(subfeature, length=feature.length) for name, subfeature in feature.feature.items()}
        if isinstance(feature, dict):
            arrays = [_e(array.field(name), subfeature) for name, subfeature in feature.items()]
            return pa.StructArray.from_arrays(arrays, names=list(feature), mask=array.is_null())
    elif pa.types.is_list(array.type):
        array_offsets = _combine_list_array_offsets_with_mask(array)
        if isinstance(feature, list):
            return pa.ListArray.from_arrays(array_offsets, _e(array.values, feature[0]))
        if isinstance(feature, Sequence) and feature.length == -1:
            return pa.ListArray.from_arrays(array_offsets, _e(array.values, feature.feature))
    elif pa.types.is_fixed_size_list(array.type):
        if isinstance(feature, Sequence) and feature.length > -1:
            array_values = array.values[array.offset * array.type.list_size:(array.offset + len(array)) * array.type.list_size]
            embedded_array_values = _e(array_values, feature.feature)
            if config.PYARROW_VERSION.major < 15:
                return pa.Array.from_buffers(pa.list_(array_values.type, feature.length), len(array), [array.is_valid().buffers()[1]], children=[embedded_array_values])
            else:
                return pa.FixedSizeListArray.from_arrays(embedded_array_values, feature.length, mask=array.is_null())
    if not isinstance(feature, (Sequence, dict, list, tuple)):
        return array
    raise TypeError(f"Couldn't embed array of type\n{array.type}\nwith\n{feature}")