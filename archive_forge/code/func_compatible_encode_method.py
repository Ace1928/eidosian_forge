import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def compatible_encode_method(bytesorstring: Union[bytes, str], encoding: str='utf-8', erraction: str='ignore') -> str:
    """When Py2 str.encode is called, it often means bytes.encode in Py3.

    This does either.
    """
    if isinstance(bytesorstring, str):
        return bytesorstring
    assert isinstance(bytesorstring, bytes), str(type(bytesorstring))
    return bytesorstring.decode(encoding, erraction)