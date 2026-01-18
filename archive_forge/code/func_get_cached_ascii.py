from codecs import utf_8_decode as _utf8_decode
from codecs import utf_8_encode as _utf8_encode
from typing import Dict
def get_cached_ascii(ascii_str, _uni_to_utf8=_unicode_to_utf8_map, _utf8_to_uni=_utf8_to_unicode_map):
    """This is a string which is identical in utf-8 and unicode."""
    uni_str = ascii_str.decode('ascii')
    ascii_str = _uni_to_utf8.setdefault(uni_str, ascii_str)
    _utf8_to_uni.setdefault(ascii_str, uni_str)
    return ascii_str