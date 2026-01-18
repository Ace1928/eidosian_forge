import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def anchorValidator(value):
    """
    Version 3+.
    """
    if not genericDictValidator(value, _anchorDictPrototype):
        return False
    x = value.get('x')
    y = value.get('y')
    if x is None or y is None:
        return False
    identifier = value.get('identifier')
    if identifier is not None and (not identifierValidator(identifier)):
        return False
    color = value.get('color')
    if color is not None and (not colorValidator(color)):
        return False
    return True