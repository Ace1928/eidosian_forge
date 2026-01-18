import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoOpenTypeOS2WidthClassValidator(value):
    """
    Version 2+.
    """
    if not isinstance(value, int):
        return False
    if value < 1:
        return False
    if value > 9:
        return False
    return True