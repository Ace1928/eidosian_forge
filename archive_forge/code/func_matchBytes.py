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
def matchBytes(self, bytes):
    """Look for a sequence of bytes at the start of a string. If the bytes
        are found return True and advance the position to the byte after the
        match. Otherwise return False and leave the position alone"""
    rv = self.startswith(bytes, self.position)
    if rv:
        self.position += len(bytes)
    return rv