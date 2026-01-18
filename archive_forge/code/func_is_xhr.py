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
@property
def is_xhr(self):
    """Is X-Requested-With header present and equal to ``XMLHttpRequest``?

        Note: this isn't set by every XMLHttpRequest request, it is
        only set if you are using a Javascript library that sets it
        (or you set the header yourself manually).  Currently
        Prototype and jQuery are known to set this header."""
    return self.environ.get('HTTP_X_REQUESTED_WITH', '') == 'XMLHttpRequest'