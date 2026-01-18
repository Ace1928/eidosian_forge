import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoWOFFMetadataUniqueIDValidator(value):
    """
    Version 3+.
    """
    dictPrototype = dict(id=(str, True))
    if not genericDictValidator(value, dictPrototype):
        return False
    return True