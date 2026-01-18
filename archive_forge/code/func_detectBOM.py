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
def detectBOM(self):
    """Attempts to detect at BOM at the start of the stream. If
        an encoding can be determined from the BOM return the name of the
        encoding otherwise return None"""
    bomDict = {codecs.BOM_UTF8: 'utf-8', codecs.BOM_UTF16_LE: 'utf-16le', codecs.BOM_UTF16_BE: 'utf-16be', codecs.BOM_UTF32_LE: 'utf-32le', codecs.BOM_UTF32_BE: 'utf-32be'}
    string = self.rawStream.read(4)
    assert isinstance(string, bytes)
    encoding = bomDict.get(string[:3])
    seek = 3
    if not encoding:
        encoding = bomDict.get(string)
        seek = 4
        if not encoding:
            encoding = bomDict.get(string[:2])
            seek = 2
    if encoding:
        self.rawStream.seek(seek)
        return lookupEncoding(encoding)
    else:
        self.rawStream.seek(0)
        return None