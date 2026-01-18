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
def handlePossibleTag(self, endTag):
    data = self.data
    if data.currentByte not in asciiLettersBytes:
        if endTag:
            data.previous()
            self.handleOther()
        return True
    c = data.skipUntil(spacesAngleBrackets)
    if c == b'<':
        data.previous()
    else:
        attr = self.getAttribute()
        while attr is not None:
            attr = self.getAttribute()
    return True