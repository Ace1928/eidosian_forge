from codecs import utf_8_decode as _utf8_decode
from codecs import utf_8_encode as _utf8_encode
from typing import Dict
def get_cached_unicode(unicode_str):
    """Return a cached version of the unicode string.

    This has a similar idea to that of intern() in that it tries
    to return a singleton string. Only it works for unicode strings.
    """
    return decode(encode(unicode_str))