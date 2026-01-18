import os
import string
import urllib.parse
import urllib.request
from typing import Optional
from .compat import WINDOWS
def path_to_url(path: str) -> str:
    """
    Convert a path to a file: URL.  The path will be made absolute and have
    quoted path parts.
    """
    path = os.path.normpath(os.path.abspath(path))
    url = urllib.parse.urljoin('file:', urllib.request.pathname2url(path))
    return url