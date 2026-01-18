from __future__ import annotations
import io
from os import PathLike
from typing import (
import warnings
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import is_list_like
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import (
from pandas.io.parsers import TextParser
def get_data_from_filepath(filepath_or_buffer: FilePath | bytes | ReadBuffer[bytes] | ReadBuffer[str], encoding: str | None, compression: CompressionOptions, storage_options: StorageOptions) -> str | bytes | ReadBuffer[bytes] | ReadBuffer[str]:
    """
    Extract raw XML data.

    The method accepts three input types:
        1. filepath (string-like)
        2. file-like object (e.g. open file object, StringIO)
        3. XML string or bytes

    This method turns (1) into (2) to simplify the rest of the processing.
    It returns input types (2) and (3) unchanged.
    """
    if not isinstance(filepath_or_buffer, bytes):
        filepath_or_buffer = stringify_path(filepath_or_buffer)
    if (isinstance(filepath_or_buffer, str) and (not filepath_or_buffer.startswith(('<?xml', '<')))) and (not isinstance(filepath_or_buffer, str) or is_url(filepath_or_buffer) or is_fsspec_url(filepath_or_buffer) or file_exists(filepath_or_buffer)):
        with get_handle(filepath_or_buffer, 'r', encoding=encoding, compression=compression, storage_options=storage_options) as handle_obj:
            filepath_or_buffer = handle_obj.handle.read() if hasattr(handle_obj.handle, 'read') else handle_obj.handle
    return filepath_or_buffer