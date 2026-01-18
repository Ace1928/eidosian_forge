import base64
import codecs
import os
import posixpath
import re
import string
from typing import (
from urllib.parse import (
from urllib.parse import _coerce_args  # type: ignore
from urllib.request import pathname2url, url2pathname
from .util import to_unicode
from ._infra import _ASCII_TAB_OR_NEWLINE, _C0_CONTROL_OR_SPACE
from ._types import AnyUnicodeError, StrOrBytes
from ._url import _SPECIAL_SCHEMES
def path_to_file_uri(path: str) -> str:
    """Convert local filesystem path to legal File URIs as described in:
    http://en.wikipedia.org/wiki/File_URI_scheme
    """
    x = pathname2url(os.path.abspath(path))
    return f'file:///{x.lstrip('/')}'