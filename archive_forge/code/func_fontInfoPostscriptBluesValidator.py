import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoPostscriptBluesValidator(values):
    """
    Version 2+.
    """
    if not isinstance(values, (list, tuple)):
        return False
    if len(values) > 14:
        return False
    if len(values) % 2:
        return False
    for value in values:
        if not isinstance(value, numberTypes):
            return False
    return True