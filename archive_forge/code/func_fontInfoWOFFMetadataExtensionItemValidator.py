import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoWOFFMetadataExtensionItemValidator(value):
    """
    Version 3+.
    """
    dictPrototype = dict(id=(str, False), names=(list, True), values=(list, True))
    if not genericDictValidator(value, dictPrototype):
        return False
    for name in value['names']:
        if not fontInfoWOFFMetadataExtensionNameValidator(name):
            return False
    for val in value['values']:
        if not fontInfoWOFFMetadataExtensionValueValidator(val):
            return False
    return True