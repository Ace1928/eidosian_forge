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
def application_url(self):
    """
        The URL including SCRIPT_NAME (no PATH_INFO or query string)
        """
    bscript_name = bytes_(self.script_name, self.url_encoding)
    return self.host_url + url_quote(bscript_name, PATH_SAFE)