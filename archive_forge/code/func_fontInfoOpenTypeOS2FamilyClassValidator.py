import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoOpenTypeOS2FamilyClassValidator(values):
    """
    Version 2+.
    """
    if not isinstance(values, (list, tuple)):
        return False
    if len(values) != 2:
        return False
    for value in values:
        if not isinstance(value, int):
            return False
    classID, subclassID = values
    if classID < 0 or classID > 14:
        return False
    if subclassID < 0 or subclassID > 15:
        return False
    return True