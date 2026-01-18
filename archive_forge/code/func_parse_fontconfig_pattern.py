from functools import lru_cache, partial
import re
from pyparsing import (
from matplotlib import _api
@lru_cache
def parse_fontconfig_pattern(pattern):
    """
    Parse a fontconfig *pattern* into a dict that can initialize a
    `.font_manager.FontProperties` object.
    """
    parser = _make_fontconfig_parser()
    try:
        parse = parser.parseString(pattern)
    except ParseException as err:
        raise ValueError('\n' + ParseException.explain(err, 0)) from None
    parser.resetCache()
    props = {}
    if 'families' in parse:
        props['family'] = [*map(_family_unescape, parse['families'])]
    if 'sizes' in parse:
        props['size'] = [*parse['sizes']]
    for prop in parse.get('properties', []):
        if len(prop) == 1:
            if prop[0] not in _CONSTANTS:
                _api.warn_deprecated('3.7', message=f'Support for unknown constants ({prop[0]!r}) is deprecated since %(since)s and will be removed %(removal)s.')
                continue
            prop = _CONSTANTS[prop[0]]
        k, *v = prop
        props.setdefault(k, []).extend(map(_value_unescape, v))
    return props