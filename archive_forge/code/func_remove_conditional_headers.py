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
def remove_conditional_headers(self, remove_encoding=True, remove_range=True, remove_match=True, remove_modified=True):
    """
        Remove headers that make the request conditional.

        These headers can cause the response to be 304 Not Modified,
        which in some cases you may not want to be possible.

        This does not remove headers like If-Match, which are used for
        conflict detection.
        """
    check_keys = []
    if remove_range:
        check_keys += ['HTTP_IF_RANGE', 'HTTP_RANGE']
    if remove_match:
        check_keys.append('HTTP_IF_NONE_MATCH')
    if remove_modified:
        check_keys.append('HTTP_IF_MODIFIED_SINCE')
    if remove_encoding:
        check_keys.append('HTTP_ACCEPT_ENCODING')
    for key in check_keys:
        if key in self.environ:
            del self.environ[key]