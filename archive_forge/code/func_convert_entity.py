import re
from html.entities import name2codepoint
from typing import Iterable, Match, AnyStr, Optional, Pattern, Tuple, Union
from urllib.parse import urljoin
from w3lib.util import to_unicode
from w3lib.url import safe_url_string
from w3lib._types import StrOrBytes
def convert_entity(m: Match[str]) -> str:
    groups = m.groupdict()
    number = None
    if groups.get('dec'):
        number = int(groups['dec'], 10)
    elif groups.get('hex'):
        number = int(groups['hex'], 16)
    elif groups.get('named'):
        entity_name = groups['named']
        if entity_name.lower() in keep:
            return m.group(0)
        else:
            number = name2codepoint.get(entity_name) or name2codepoint.get(entity_name.lower())
    if number is not None:
        try:
            if 128 <= number <= 159:
                return bytes((number,)).decode('cp1252')
            else:
                return chr(number)
        except (ValueError, OverflowError):
            pass
    return '' if remove_illegal and groups.get('semicolon') else m.group(0)