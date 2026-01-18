import os
from io import BytesIO, StringIO, UnsupportedOperation
from django.core.files.utils import FileProxyMixin
from django.utils.functional import cached_property
def endswith_lf(line):
    """Return True if line (a text or bytestring) ends with '
'."""
    return line.endswith('\n' if isinstance(line, str) else b'\n')