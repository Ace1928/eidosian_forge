import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def fontInfoWOFFMetadataDescriptionValidator(value):
    """
    Version 3+.
    """
    dictPrototype = dict(url=(str, False), text=(list, True))
    if not genericDictValidator(value, dictPrototype):
        return False
    for text in value['text']:
        if not fontInfoWOFFMetadataTextValue(text):
            return False
    return True