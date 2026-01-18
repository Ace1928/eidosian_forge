from __future__ import division, print_function, absolute_import
import locale
import codecs
from petl.compat import izip_longest
from petl.util.base import Table
def getcodec(encoding):
    if encoding is None:
        encoding = locale.getpreferredencoding()
    codec = codecs.lookup(encoding)
    return codec