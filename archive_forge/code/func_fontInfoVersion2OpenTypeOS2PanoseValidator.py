import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoVersion2OpenTypeOS2PanoseValidator(values):
    """
    Version 2.
    """
    if not isinstance(values, (list, tuple)):
        return False
    if len(values) != 10:
        return False
    for value in values:
        if not isinstance(value, int):
            return False
    return True