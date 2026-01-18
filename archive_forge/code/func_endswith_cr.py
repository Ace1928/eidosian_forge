import os
from io import BytesIO, StringIO, UnsupportedOperation
from django.core.files.utils import FileProxyMixin
from django.utils.functional import cached_property
def endswith_cr(line):
    """Return True if line (a text or bytestring) ends with '\r'."""
    return line.endswith('\r' if isinstance(line, str) else b'\r')