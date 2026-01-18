import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoWOFFMetadataVendorValidator(value):
    """
    Version 3+.
    """
    dictPrototype = {'name': (str, True), 'url': (str, False), 'dir': (str, False), 'class': (str, False)}
    if not genericDictValidator(value, dictPrototype):
        return False
    if 'dir' in value and value.get('dir') not in ('ltr', 'rtl'):
        return False
    return True