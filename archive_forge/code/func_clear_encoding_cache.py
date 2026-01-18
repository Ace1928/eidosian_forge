from codecs import utf_8_decode as _utf8_decode
from codecs import utf_8_encode as _utf8_encode
from typing import Dict
def clear_encoding_cache():
    """Clear the encoding and decoding caches"""
    _unicode_to_utf8_map.clear()
    _utf8_to_unicode_map.clear()