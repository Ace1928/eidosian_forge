import logging
import re
from typing import Dict, Iterable, Optional, cast
from .glyphlist import glyphname2unicode
from .latin_enc import ENCODING
from .psparser import PSLiteral
def name2unicode(name: str) -> str:
    """Converts Adobe glyph names to Unicode numbers.

    In contrast to the specification, this raises a KeyError instead of return
    an empty string when the key is unknown.
    This way the caller must explicitly define what to do
    when there is not a match.

    Reference:
    https://github.com/adobe-type-tools/agl-specification#2-the-mapping

    :returns unicode character if name resembles something,
    otherwise a KeyError
    """
    if not isinstance(name, str):
        raise KeyError('Could not convert unicode name "%s" to character because it should be of type str but is of type %s' % (name, type(name)))
    name = name.split('.')[0]
    components = name.split('_')
    if len(components) > 1:
        return ''.join(map(name2unicode, components))
    elif name in glyphname2unicode:
        return glyphname2unicode[name]
    elif name.startswith('uni'):
        name_without_uni = name.strip('uni')
        if HEXADECIMAL.match(name_without_uni) and len(name_without_uni) % 4 == 0:
            unicode_digits = [int(name_without_uni[i:i + 4], base=16) for i in range(0, len(name_without_uni), 4)]
            for digit in unicode_digits:
                raise_key_error_for_invalid_unicode(digit)
            characters = map(chr, unicode_digits)
            return ''.join(characters)
    elif name.startswith('u'):
        name_without_u = name.strip('u')
        if HEXADECIMAL.match(name_without_u) and 4 <= len(name_without_u) <= 6:
            unicode_digit = int(name_without_u, base=16)
            raise_key_error_for_invalid_unicode(unicode_digit)
            return chr(unicode_digit)
    raise KeyError('Could not convert unicode name "%s" to character because it does not match specification' % name)