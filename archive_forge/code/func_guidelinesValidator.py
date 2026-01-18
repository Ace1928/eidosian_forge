import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def guidelinesValidator(value, identifiers=None):
    """
    Version 3+.
    """
    if not isinstance(value, list):
        return False
    if identifiers is None:
        identifiers = set()
    for guide in value:
        if not guidelineValidator(guide):
            return False
        identifier = guide.get('identifier')
        if identifier is not None:
            if identifier in identifiers:
                return False
            identifiers.add(identifier)
    return True