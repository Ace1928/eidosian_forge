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
def copy_get(self):
    """
        Copies the request and environment object, but turning this request
        into a GET along the way.  If this was a POST request (or any other
        verb) then it becomes GET, and the request body is thrown away.
        """
    env = self.environ.copy()
    return self.__class__(env, method='GET', content_type=None, body=b'')