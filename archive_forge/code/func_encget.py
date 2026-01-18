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
def encget(self, key, default=NoDefault, encattr=None):
    val = self.environ.get(key, default)
    if val is NoDefault:
        raise KeyError(key)
    if val is default:
        return default
    return val