import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoWOFFMetadataCreditsValidator(value):
    """
    Version 3+.
    """
    dictPrototype = dict(credits=(list, True))
    if not genericDictValidator(value, dictPrototype):
        return False
    if not len(value['credits']):
        return False
    dictPrototype = {'name': (str, True), 'url': (str, False), 'role': (str, False), 'dir': (str, False), 'class': (str, False)}
    for credit in value['credits']:
        if not genericDictValidator(credit, dictPrototype):
            return False
        if 'dir' in credit and credit.get('dir') not in ('ltr', 'rtl'):
            return False
    return True