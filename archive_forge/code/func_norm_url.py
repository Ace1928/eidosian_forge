from __future__ import annotations
import pathlib
from typing import IO, TYPE_CHECKING, Any, Optional, TextIO, Tuple, Union
from io import TextIOBase, TextIOWrapper
from posixpath import normpath, sep
from urllib.parse import urljoin, urlsplit, urlunsplit
from rdflib.parser import (
def norm_url(base: str, url: str) -> str:
    """
    >>> norm_url('http://example.org/', '/one')
    'http://example.org/one'
    >>> norm_url('http://example.org/', '/one#')
    'http://example.org/one#'
    >>> norm_url('http://example.org/one', 'two')
    'http://example.org/two'
    >>> norm_url('http://example.org/one/', 'two')
    'http://example.org/one/two'
    >>> norm_url('http://example.org/', 'http://example.net/one')
    'http://example.net/one'
    >>> norm_url('http://example.org/', 'http://example.org//one')
    'http://example.org//one'
    """
    if '://' in url:
        return url
    parts = urlsplit(urljoin(base, url))
    path = normpath(parts[2])
    if sep != '/':
        path = '/'.join(path.split(sep))
    if parts[2].endswith('/') and (not path.endswith('/')):
        path += '/'
    result = urlunsplit(parts[0:2] + (path,) + parts[3:])
    if url.endswith('#') and (not result.endswith('#')):
        result += '#'
    return result