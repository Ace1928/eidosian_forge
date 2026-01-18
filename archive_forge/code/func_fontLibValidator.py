import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontLibValidator(value):
    """
    Check the validity of the lib.
    Version 3+ (though it's backwards compatible with UFO 1 and UFO 2).

    >>> lib = {"foo" : "bar"}
    >>> fontLibValidator(lib)
    (True, None)

    >>> lib = {"public.awesome" : "hello"}
    >>> fontLibValidator(lib)
    (True, None)

    >>> lib = {"public.glyphOrder" : ["A", "C", "B"]}
    >>> fontLibValidator(lib)
    (True, None)

    >>> lib = "hello"
    >>> valid, msg = fontLibValidator(lib)
    >>> valid
    False
    >>> print(msg)  # doctest: +ELLIPSIS
    The lib data is not in the correct format: expected a dictionary, ...

    >>> lib = {1: "hello"}
    >>> valid, msg = fontLibValidator(lib)
    >>> valid
    False
    >>> print(msg)
    The lib key is not properly formatted: expected str, found int: 1

    >>> lib = {"public.glyphOrder" : "hello"}
    >>> valid, msg = fontLibValidator(lib)
    >>> valid
    False
    >>> print(msg)  # doctest: +ELLIPSIS
    public.glyphOrder is not properly formatted: expected list or tuple,...

    >>> lib = {"public.glyphOrder" : ["A", 1, "B"]}
    >>> valid, msg = fontLibValidator(lib)
    >>> valid
    False
    >>> print(msg)  # doctest: +ELLIPSIS
    public.glyphOrder is not properly formatted: expected str,...
    """
    if not isDictEnough(value):
        reason = 'expected a dictionary, found %s' % type(value).__name__
        return (False, _bogusLibFormatMessage % reason)
    for key, value in value.items():
        if not isinstance(key, str):
            return (False, 'The lib key is not properly formatted: expected str, found %s: %r' % (type(key).__name__, key))
        if key == 'public.glyphOrder':
            bogusGlyphOrderMessage = 'public.glyphOrder is not properly formatted: %s'
            if not isinstance(value, (list, tuple)):
                reason = 'expected list or tuple, found %s' % type(value).__name__
                return (False, bogusGlyphOrderMessage % reason)
            for glyphName in value:
                if not isinstance(glyphName, str):
                    reason = 'expected str, found %s' % type(glyphName).__name__
                    return (False, bogusGlyphOrderMessage % reason)
    return (True, None)