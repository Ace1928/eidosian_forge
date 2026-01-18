import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def guidelineValidator(value):
    """
    Version 3+.
    """
    if not genericDictValidator(value, _guidelineDictPrototype):
        return False
    x = value.get('x')
    y = value.get('y')
    angle = value.get('angle')
    if x is None and y is None:
        return False
    if x is None or y is None:
        if angle is not None:
            return False
    if x is not None and y is not None and (angle is None):
        return False
    if angle is not None:
        if angle < 0:
            return False
        if angle > 360:
            return False
    identifier = value.get('identifier')
    if identifier is not None and (not identifierValidator(identifier)):
        return False
    color = value.get('color')
    if color is not None and (not colorValidator(color)):
        return False
    return True