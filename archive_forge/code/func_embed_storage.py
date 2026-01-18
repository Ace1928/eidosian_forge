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
def embed_storage(self, storage: pa.StructArray) -> pa.StructArray:
    """Embed image files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.

        Returns:
            `pa.StructArray`: Array in the Image arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """

    @no_op_if_value_is_null
    def path_to_bytes(path):
        with xopen(path, 'rb') as f:
            bytes_ = f.read()
        return bytes_
    bytes_array = pa.array([(path_to_bytes(x['path']) if x['bytes'] is None else x['bytes']) if x is not None else None for x in storage.to_pylist()], type=pa.binary())
    path_array = pa.array([os.path.basename(path) if path is not None else None for path in storage.field('path').to_pylist()], type=pa.string())
    storage = pa.StructArray.from_arrays([bytes_array, path_array], ['bytes', 'path'], mask=bytes_array.is_null())
    return array_cast(storage, self.pa_type)