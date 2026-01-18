import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def identifierValidator(value):
    """
    Version 3+.

    >>> identifierValidator("a")
    True
    >>> identifierValidator("")
    False
    >>> identifierValidator("a" * 101)
    False
    """
    validCharactersMin = 32
    validCharactersMax = 126
    if not isinstance(value, str):
        return False
    if not value:
        return False
    if len(value) > 100:
        return False
    for c in value:
        c = ord(c)
        if c < validCharactersMin or c > validCharactersMax:
            return False
    return True