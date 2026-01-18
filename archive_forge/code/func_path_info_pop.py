import binascii
import io
import os
import re
import sys
import tempfile
import mimetypes
import warnings
from webob.acceptparse import (
from webob.cachecontrol import (
from webob.compat import (
from webob.cookies import RequestCookies
from webob.descriptors import (
from webob.etag import (
from webob.headers import EnvironHeaders
from webob.multidict import (
def path_info_pop(self, pattern=None):
    """
        'Pops' off the next segment of PATH_INFO, pushing it onto
        SCRIPT_NAME, and returning the popped segment.  Returns None if
        there is nothing left on PATH_INFO.

        Does not return ``''`` when there's an empty segment (like
        ``/path//path``); these segments are just ignored.

        Optional ``pattern`` argument is a regexp to match the return value
        before returning. If there is no match, no changes are made to the
        request and None is returned.
        """
    path = self.path_info
    if not path:
        return None
    slashes = ''
    while path.startswith('/'):
        slashes += '/'
        path = path[1:]
    idx = path.find('/')
    if idx == -1:
        idx = len(path)
    r = path[:idx]
    if pattern is None or re.match(pattern, r):
        self.script_name += slashes + r
        self.path_info = path[idx:]
        return r