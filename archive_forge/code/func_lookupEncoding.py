from __future__ import absolute_import, division, unicode_literals
from six import text_type
from six.moves import http_client, urllib
import codecs
import re
from io import BytesIO, StringIO
from tensorboard._vendor import webencodings
from .constants import EOF, spaceCharacters, asciiLetters, asciiUppercase
from .constants import _ReparseException
from . import _utils
def lookupEncoding(encoding):
    """Return the python codec name corresponding to an encoding or None if the
    string doesn't correspond to a valid encoding."""
    if isinstance(encoding, bytes):
        try:
            encoding = encoding.decode('ascii')
        except UnicodeDecodeError:
            return None
    if encoding is not None:
        try:
            return webencodings.lookup(encoding)
        except AttributeError:
            return None
    else:
        return None