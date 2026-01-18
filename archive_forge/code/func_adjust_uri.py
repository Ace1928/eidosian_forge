import os
import posixpath
import re
import stat
import threading
from mako import exceptions
from mako import util
from mako.template import Template
def adjust_uri(self, uri, relativeto):
    """Adjust the given ``uri`` based on the given relative URI."""
    key = (uri, relativeto)
    if key in self._uri_cache:
        return self._uri_cache[key]
    if uri[0] == '/':
        v = self._uri_cache[key] = uri
    elif relativeto is not None:
        v = self._uri_cache[key] = posixpath.join(posixpath.dirname(relativeto), uri)
    else:
        v = self._uri_cache[key] = '/' + uri
    return v