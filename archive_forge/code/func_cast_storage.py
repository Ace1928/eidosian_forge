import os
import sys
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union
import numpy as np
import pyarrow as pa
from .. import config
from ..download.download_config import DownloadConfig
from ..download.streaming_download_manager import xopen
from ..table import array_cast
from ..utils.file_utils import is_local_path
from ..utils.py_utils import first_non_null_value, no_op_if_value_is_null, string_to_dict
def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray, pa.ListArray]) -> pa.StructArray:
    """Cast an Arrow array to the Image arrow storage type.
        The Arrow types that can be converted to the Image pyarrow storage type are:

        - `pa.string()` - it must contain the "path" data
        - `pa.binary()` - it must contain the image bytes
        - `pa.struct({"bytes": pa.binary()})`
        - `pa.struct({"path": pa.string()})`
        - `pa.struct({"bytes": pa.binary(), "path": pa.string()})`  - order doesn't matter
        - `pa.list(*)` - it must contain the image array data

        Args:
            storage (`Union[pa.StringArray, pa.StructArray, pa.ListArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the Image arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
    if pa.types.is_string(storage.type):
        bytes_array = pa.array([None] * len(storage), type=pa.binary())
        storage = pa.StructArray.from_arrays([bytes_array, storage], ['bytes', 'path'], mask=storage.is_null())
    elif pa.types.is_binary(storage.type):
        path_array = pa.array([None] * len(storage), type=pa.string())
        storage = pa.StructArray.from_arrays([storage, path_array], ['bytes', 'path'], mask=storage.is_null())
    elif pa.types.is_struct(storage.type):
        if storage.type.get_field_index('bytes') >= 0:
            bytes_array = storage.field('bytes')
        else:
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
        if storage.type.get_field_index('path') >= 0:
            path_array = storage.field('path')
        else:
            path_array = pa.array([None] * len(storage), type=pa.string())
        storage = pa.StructArray.from_arrays([bytes_array, path_array], ['bytes', 'path'], mask=storage.is_null())
    elif pa.types.is_list(storage.type):
        bytes_array = pa.array([encode_np_array(np.array(arr))['bytes'] if arr is not None else None for arr in storage.to_pylist()], type=pa.binary())
        path_array = pa.array([None] * len(storage), type=pa.string())
        storage = pa.StructArray.from_arrays([bytes_array, path_array], ['bytes', 'path'], mask=bytes_array.is_null())
    return array_cast(storage, self.pa_type)