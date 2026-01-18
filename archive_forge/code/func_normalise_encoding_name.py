from __future__ import absolute_import
import os
from .. import Utils
def normalise_encoding_name(option_name, encoding):
    """
    >>> normalise_encoding_name('c_string_encoding', 'ascii')
    'ascii'
    >>> normalise_encoding_name('c_string_encoding', 'AsCIi')
    'ascii'
    >>> normalise_encoding_name('c_string_encoding', 'us-ascii')
    'ascii'
    >>> normalise_encoding_name('c_string_encoding', 'utF8')
    'utf8'
    >>> normalise_encoding_name('c_string_encoding', 'utF-8')
    'utf8'
    >>> normalise_encoding_name('c_string_encoding', 'deFAuLT')
    'default'
    >>> normalise_encoding_name('c_string_encoding', 'default')
    'default'
    >>> normalise_encoding_name('c_string_encoding', 'SeriousLyNoSuch--Encoding')
    'SeriousLyNoSuch--Encoding'
    """
    if not encoding:
        return ''
    if encoding.lower() in ('default', 'ascii', 'utf8'):
        return encoding.lower()
    import codecs
    try:
        decoder = codecs.getdecoder(encoding)
    except LookupError:
        return encoding
    for name in ('ascii', 'utf8'):
        if codecs.getdecoder(name) == decoder:
            return name
    return encoding