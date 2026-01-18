import os
from io import BytesIO, StringIO, UnsupportedOperation
from django.core.files.utils import FileProxyMixin
from django.utils.functional import cached_property
def equals_lf(line):
    """Return True if line (a text or bytestring) equals '
'."""
    return line == ('\n' if isinstance(line, str) else b'\n')