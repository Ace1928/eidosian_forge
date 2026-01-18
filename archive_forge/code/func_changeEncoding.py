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
def changeEncoding(self, newEncoding):
    assert self.charEncoding[1] != 'certain'
    newEncoding = lookupEncoding(newEncoding)
    if newEncoding is None:
        return
    if newEncoding.name in ('utf-16be', 'utf-16le'):
        newEncoding = lookupEncoding('utf-8')
        assert newEncoding is not None
    elif newEncoding == self.charEncoding[0]:
        self.charEncoding = (self.charEncoding[0], 'certain')
    else:
        self.rawStream.seek(0)
        self.charEncoding = (newEncoding, 'certain')
        self.reset()
        raise _ReparseException('Encoding changed from %s to %s' % (self.charEncoding[0], newEncoding))