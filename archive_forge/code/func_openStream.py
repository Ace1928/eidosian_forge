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
def openStream(self, source):
    """Produces a file object from source.

        source can be either a file object, local filename or a string.

        """
    if hasattr(source, 'read'):
        stream = source
    else:
        stream = BytesIO(source)
    try:
        stream.seek(stream.tell())
    except Exception:
        stream = BufferedStream(stream)
    return stream