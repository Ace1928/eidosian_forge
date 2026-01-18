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
def safe_download_url(url: StrOrBytes, encoding: str='utf8', path_encoding: str='utf8') -> str:
    """Make a url for download. This will call safe_url_string
    and then strip the fragment, if one exists. The path will
    be normalised.

    If the path is outside the document root, it will be changed
    to be within the document root.
    """
    safe_url = safe_url_string(url, encoding, path_encoding)
    scheme, netloc, path, query, _ = urlsplit(safe_url)
    if path:
        path = _parent_dirs.sub('', posixpath.normpath(path))
        if safe_url.endswith('/') and (not path.endswith('/')):
            path += '/'
    else:
        path = '/'
    return urlunsplit((scheme, netloc, path, query, ''))