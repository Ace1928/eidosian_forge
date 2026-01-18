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
def cast_array_to_feature(array: pa.Array, feature: 'FeatureType', allow_number_to_str=True):
    """Cast an array to the arrow type that corresponds to the requested feature type.
    For custom features like [`Audio`] or [`Image`], it takes into account the "cast_storage" methods
    they defined to enable casting from other arrow types.

    Args:
        array (`pa.Array`):
            The PyArrow array to cast.
        feature (`datasets.features.FeatureType`):
            The target feature type.
        allow_number_to_str (`bool`, defaults to `True`):
            Whether to allow casting numbers to strings.
            Defaults to `True`.

    Raises:
        `pa.ArrowInvalidError`: if the arrow data casting fails
        `TypeError`: if the target type is not supported according, e.g.

            - if a field is missing
            - if casting from numbers to strings and `allow_number_to_str` is `False`

    Returns:
        array (`pyarrow.Array`): the casted array
    """
    from .features.features import Sequence, get_nested_type
    _c = partial(cast_array_to_feature, allow_number_to_str=allow_number_to_str)
    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if hasattr(feature, 'cast_storage'):
        return feature.cast_storage(array)
    elif pa.types.is_struct(array.type):
        if isinstance(feature, Sequence) and isinstance(feature.feature, dict):
            feature = {name: Sequence(subfeature, length=feature.length) for name, subfeature in feature.feature.items()}
        if isinstance(feature, dict) and {field.name for field in array.type} == set(feature):
            if array.type.num_fields == 0:
                return array
            arrays = [_c(array.field(name), subfeature) for name, subfeature in feature.items()]
            return pa.StructArray.from_arrays(arrays, names=list(feature), mask=array.is_null())
    elif pa.types.is_list(array.type):
        if isinstance(feature, list):
            casted_array_values = _c(array.values, feature[0])
            if casted_array_values.type == array.values.type:
                return array
            else:
                array_offsets = _combine_list_array_offsets_with_mask(array)
                return pa.ListArray.from_arrays(array_offsets, casted_array_values)
        elif isinstance(feature, Sequence):
            if feature.length > -1:
                if _are_list_values_of_length(array, feature.length):
                    if array.null_count > 0:
                        array_type = array.type
                        storage_type = _storage_type(array_type)
                        if array_type != storage_type:
                            array = array_cast(array, storage_type, allow_number_to_str=allow_number_to_str)
                            array = pc.list_slice(array, 0, feature.length, return_fixed_size_list=True)
                            array = array_cast(array, array_type, allow_number_to_str=allow_number_to_str)
                        else:
                            array = pc.list_slice(array, 0, feature.length, return_fixed_size_list=True)
                        array_values = array.values
                        casted_array_values = _c(array_values, feature.feature)
                        if config.PYARROW_VERSION.major < 15:
                            return pa.Array.from_buffers(pa.list_(casted_array_values.type, feature.length), len(array), [array.is_valid().buffers()[1]], children=[casted_array_values])
                        else:
                            return pa.FixedSizeListArray.from_arrays(casted_array_values, feature.length, mask=array.is_null())
                    else:
                        array_values = array.values[array.offset * feature.length:(array.offset + len(array)) * feature.length]
                        return pa.FixedSizeListArray.from_arrays(_c(array_values, feature.feature), feature.length)
            else:
                casted_array_values = _c(array.values, feature.feature)
                if casted_array_values.type == array.values.type:
                    return array
                else:
                    array_offsets = _combine_list_array_offsets_with_mask(array)
                    return pa.ListArray.from_arrays(array_offsets, casted_array_values)
    elif pa.types.is_fixed_size_list(array.type):
        if isinstance(feature, list):
            array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
            return pa.ListArray.from_arrays(array_offsets, _c(array.values, feature[0]), mask=array.is_null())
        elif isinstance(feature, Sequence):
            if feature.length > -1:
                if feature.length == array.type.list_size:
                    array_values = array.values[array.offset * array.type.list_size:(array.offset + len(array)) * array.type.list_size]
                    casted_array_values = _c(array_values, feature.feature)
                    if config.PYARROW_VERSION.major < 15:
                        return pa.Array.from_buffers(pa.list_(casted_array_values.type, feature.length), len(array), [array.is_valid().buffers()[1]], children=[casted_array_values])
                    else:
                        return pa.FixedSizeListArray.from_arrays(casted_array_values, feature.length, mask=array.is_null())
            else:
                array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
                return pa.ListArray.from_arrays(array_offsets, _c(array.values, feature.feature), mask=array.is_null())
    if pa.types.is_null(array.type):
        return array_cast(array, get_nested_type(feature), allow_number_to_str=allow_number_to_str)
    elif not isinstance(feature, (Sequence, dict, list, tuple)):
        return array_cast(array, feature(), allow_number_to_str=allow_number_to_str)
    raise TypeError(f"Couldn't cast array of type\n{array.type}\nto\n{feature}")