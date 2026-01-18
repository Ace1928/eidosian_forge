import copy
import time
import calendar
from ._internal_utils import to_native_string
from .compat import cookielib, urlparse, urlunparse, Morsel, MutableMapping
def add_unredirected_header(self, name, value):
    self._new_headers[name] = value