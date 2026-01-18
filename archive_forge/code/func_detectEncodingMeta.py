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
def detectEncodingMeta(self):
    """Report the encoding declared by the meta element
        """
    buffer = self.rawStream.read(self.numBytesMeta)
    assert isinstance(buffer, bytes)
    parser = EncodingParser(buffer)
    self.rawStream.seek(0)
    encoding = parser.getEncoding()
    if encoding is not None and encoding.name in ('utf-16be', 'utf-16le'):
        encoding = lookupEncoding('utf-8')
    return encoding