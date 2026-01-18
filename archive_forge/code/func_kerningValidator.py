import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def kerningValidator(data):
    """
    Check the validity of the kerning data structure.
    Version 3+ (though it's backwards compatible with UFO 1 and UFO 2).

    >>> kerning = {"A" : {"B" : 100}}
    >>> kerningValidator(kerning)
    (True, None)

    >>> kerning = {"A" : ["B"]}
    >>> valid, msg = kerningValidator(kerning)
    >>> valid
    False
    >>> print(msg)
    The kerning data is not in the correct format.

    >>> kerning = {"A" : {"B" : "100"}}
    >>> valid, msg = kerningValidator(kerning)
    >>> valid
    False
    >>> print(msg)
    The kerning data is not in the correct format.
    """
    bogusFormatMessage = 'The kerning data is not in the correct format.'
    if not isinstance(data, Mapping):
        return (False, bogusFormatMessage)
    for first, secondDict in data.items():
        if not isinstance(first, str):
            return (False, bogusFormatMessage)
        elif not isinstance(secondDict, Mapping):
            return (False, bogusFormatMessage)
        for second, value in secondDict.items():
            if not isinstance(second, str):
                return (False, bogusFormatMessage)
            elif not isinstance(value, numberTypes):
                return (False, bogusFormatMessage)
    return (True, None)