import os
from copy import deepcopy
from os import fsdecode
import logging
import zipfile
import enum
from collections import OrderedDict
import fs
import fs.base
import fs.subfs
import fs.errors
import fs.copy
import fs.osfs
import fs.zipfs
import fs.tempfs
import fs.tools
from fontTools.misc import plistlib
from fontTools.ufoLib.validators import *
from fontTools.ufoLib.filenames import userNameToFileName
from fontTools.ufoLib.converters import convertUFO1OrUFO2KerningToUFO3Kerning
from fontTools.ufoLib.errors import UFOLibError
from fontTools.ufoLib.utils import numberTypes, _VersionTupleEnumMixin
def convertFontInfoValueForAttributeFromVersion1ToVersion2(attr, value):
    """
    Convert value from version 1 to version 2 format.
    Returns the new attribute name and the converted value.
    If the value is None, None will be returned for the new value.
    """
    if isinstance(value, float):
        if int(value) == value:
            value = int(value)
    if value is not None:
        if attr == 'fontStyle':
            v = _fontStyle1To2.get(value)
            if v is None:
                raise UFOLibError(f'Cannot convert value ({value!r}) for attribute {attr}.')
            value = v
        elif attr == 'widthName':
            v = _widthName1To2.get(value)
            if v is None:
                raise UFOLibError(f'Cannot convert value ({value!r}) for attribute {attr}.')
            value = v
        elif attr == 'msCharSet':
            v = _msCharSet1To2.get(value)
            if v is None:
                raise UFOLibError(f'Cannot convert value ({value!r}) for attribute {attr}.')
            value = v
    attr = fontInfoAttributesVersion1To2.get(attr, attr)
    return (attr, value)