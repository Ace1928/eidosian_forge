import re
from typing import Dict, Union
from xml.etree.ElementTree import (Element, ElementTree, ParseError,
from .. import errors, lazy_regex
from . import inventory, serializer
def encode_and_escape(unicode_or_utf8_str, _map=_to_escaped_map):
    """Encode the string into utf8, and escape invalid XML characters"""
    text = _map.get(unicode_or_utf8_str)
    if text is None:
        if isinstance(unicode_or_utf8_str, str):
            text = _unicode_re.sub(_unicode_escape_replace, unicode_or_utf8_str).encode()
        else:
            text = _utf8_re.sub(_utf8_escape_replace, unicode_or_utf8_str)
        _map[unicode_or_utf8_str] = text
    return text