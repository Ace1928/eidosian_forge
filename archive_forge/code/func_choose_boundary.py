from __future__ import absolute_import
import binascii
import codecs
import os
from io import BytesIO
from .fields import RequestField
from .packages import six
from .packages.six import b
def choose_boundary():
    """
    Our embarrassingly-simple replacement for mimetools.choose_boundary.
    """
    boundary = binascii.hexlify(os.urandom(16))
    if not six.PY2:
        boundary = boundary.decode('ascii')
    return boundary