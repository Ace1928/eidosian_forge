import copy
import time
import calendar
from ._internal_utils import to_native_string
from .compat import cookielib, urlparse, urlunparse, Morsel, MutableMapping
def list_paths(self):
    """Utility method to list all the paths in the jar."""
    paths = []
    for cookie in iter(self):
        if cookie.path not in paths:
            paths.append(cookie.path)
    return paths