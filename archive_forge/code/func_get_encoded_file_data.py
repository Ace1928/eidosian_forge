from __future__ import annotations
import contextlib
import errno
import io
import os
from pathlib import Path
from streamlit import env_util, util
from streamlit.string_util import is_binary_string
def get_encoded_file_data(data: bytes, encoding: str='auto') -> io.StringIO | io.BytesIO:
    """Coerce bytes to a BytesIO or a StringIO.

    Parameters
    ----------
    data : bytes
    encoding : str

    Returns
    -------
    BytesIO or StringIO
        If the file's data is in a well-known textual format (or if the encoding
        parameter is set), return a StringIO. Otherwise, return BytesIO.

    """
    if encoding == 'auto':
        data_encoding = None if is_binary_string(data) else 'utf-8'
    else:
        data_encoding = encoding
    if data_encoding:
        return io.StringIO(data.decode(data_encoding))
    return io.BytesIO(data)