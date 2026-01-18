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
def determineEncoding(self, chardet=True):
    charEncoding = (self.detectBOM(), 'certain')
    if charEncoding[0] is not None:
        return charEncoding
    charEncoding = (lookupEncoding(self.override_encoding), 'certain')
    if charEncoding[0] is not None:
        return charEncoding
    charEncoding = (lookupEncoding(self.transport_encoding), 'certain')
    if charEncoding[0] is not None:
        return charEncoding
    charEncoding = (self.detectEncodingMeta(), 'tentative')
    if charEncoding[0] is not None:
        return charEncoding
    charEncoding = (lookupEncoding(self.same_origin_parent_encoding), 'tentative')
    if charEncoding[0] is not None and (not charEncoding[0].name.startswith('utf-16')):
        return charEncoding
    charEncoding = (lookupEncoding(self.likely_encoding), 'tentative')
    if charEncoding[0] is not None:
        return charEncoding
    if chardet:
        try:
            from chardet.universaldetector import UniversalDetector
        except ImportError:
            pass
        else:
            buffers = []
            detector = UniversalDetector()
            while not detector.done:
                buffer = self.rawStream.read(self.numBytesChardet)
                assert isinstance(buffer, bytes)
                if not buffer:
                    break
                buffers.append(buffer)
                detector.feed(buffer)
            detector.close()
            encoding = lookupEncoding(detector.result['encoding'])
            self.rawStream.seek(0)
            if encoding is not None:
                return (encoding, 'tentative')
    charEncoding = (lookupEncoding(self.default_encoding), 'tentative')
    if charEncoding[0] is not None:
        return charEncoding
    return (lookupEncoding('windows-1252'), 'tentative')