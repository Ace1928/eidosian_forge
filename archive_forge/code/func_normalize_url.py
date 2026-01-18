import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def normalize_url(url):
    """Make sure that a path string is in fully normalized URL form.

    This handles URLs which have unicode characters, spaces,
    special characters, etc.

    It has two basic modes of operation, depending on whether the
    supplied string starts with a url specifier (scheme://) or not.
    If it does not have a specifier it is considered a local path,
    and will be converted into a file:/// url. Non-ascii characters
    will be encoded using utf-8.
    If it does have a url specifier, it will be treated as a "hybrid"
    URL. Basically, a URL that should have URL special characters already
    escaped (like +?&# etc), but may have unicode characters, etc
    which would not be valid in a real URL.

    Args:
      url: Either a hybrid URL or a local path
    Returns: A normalized URL which only includes 7-bit ASCII characters.
    """
    scheme_end, path_start = _find_scheme_and_separator(url)
    if scheme_end is None:
        return local_path_to_url(url)
    prefix = url[:path_start]
    path = url[path_start:]
    if not isinstance(url, str):
        for c in url:
            if c not in _url_safe_characters:
                raise InvalidURL(url, 'URLs can only contain specific safe characters (not %r)' % c)
        path = _url_hex_escapes_re.sub(_unescape_safe_chars, path)
        return str(prefix + ''.join(path))
    path_chars = list(path)
    for i in range(len(path_chars)):
        if path_chars[i] not in _url_safe_characters:
            path_chars[i] = ''.join(['%%%02X' % c for c in bytearray(path_chars[i].encode('utf-8'))])
    path = ''.join(path_chars)
    path = _url_hex_escapes_re.sub(_unescape_safe_chars, path)
    return str(prefix + path)