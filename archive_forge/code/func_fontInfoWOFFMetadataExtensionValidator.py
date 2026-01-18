import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoWOFFMetadataExtensionValidator(value):
    """
    Version 3+.
    """
    dictPrototype = dict(names=(list, False), items=(list, True), id=(str, False))
    if not genericDictValidator(value, dictPrototype):
        return False
    if 'names' in value:
        for name in value['names']:
            if not fontInfoWOFFMetadataExtensionNameValidator(name):
                return False
    for item in value['items']:
        if not fontInfoWOFFMetadataExtensionItemValidator(item):
            return False
    return True