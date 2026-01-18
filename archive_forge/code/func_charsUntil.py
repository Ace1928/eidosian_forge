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
def charsUntil(self, characters, opposite=False):
    """ Returns a string of characters from the stream up to but not
        including any character in 'characters' or EOF. 'characters' must be
        a container that supports the 'in' method and iteration over its
        characters.
        """
    try:
        chars = charsUntilRegEx[characters, opposite]
    except KeyError:
        if __debug__:
            for c in characters:
                assert ord(c) < 128
        regex = ''.join(['\\x%02x' % ord(c) for c in characters])
        if not opposite:
            regex = '^%s' % regex
        chars = charsUntilRegEx[characters, opposite] = re.compile('[%s]+' % regex)
    rv = []
    while True:
        m = chars.match(self.chunk, self.chunkOffset)
        if m is None:
            if self.chunkOffset != self.chunkSize:
                break
        else:
            end = m.end()
            if end != self.chunkSize:
                rv.append(self.chunk[self.chunkOffset:end])
                self.chunkOffset = end
                break
        rv.append(self.chunk[self.chunkOffset:])
        if not self.readChunk():
            break
    r = ''.join(rv)
    return r