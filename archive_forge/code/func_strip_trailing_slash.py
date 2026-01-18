import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def strip_trailing_slash(url):
    """Strip trailing slash, except for root paths.

    The definition of 'root path' is platform-dependent.
    This assumes that all URLs are valid netloc urls, such that they
    form:
    scheme://host/path
    It searches for ://, and then refuses to remove the next '/'.
    It can also handle relative paths
    Examples:
        path/to/foo       => path/to/foo
        path/to/foo/      => path/to/foo
        http://host/path/ => http://host/path
        http://host/path  => http://host/path
        http://host/      => http://host/
        file:///          => file:///
        file:///foo/      => file:///foo
        # This is unique on win32 platforms, and is the only URL
        # format which does it differently.
        file:///c|/       => file:///c:/
    """
    if not url.endswith('/'):
        return url
    if sys.platform == 'win32' and url.startswith('file://'):
        return _win32_strip_local_trailing_slash(url)
    scheme_loc, first_path_slash = _find_scheme_and_separator(url)
    if scheme_loc is None:
        return url[:-1]
    if first_path_slash is None or first_path_slash == len(url) - 1:
        return url
    return url[:-1]