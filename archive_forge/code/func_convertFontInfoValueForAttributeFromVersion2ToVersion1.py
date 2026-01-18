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
def convertFontInfoValueForAttributeFromVersion2ToVersion1(attr, value):
    """
    Convert value from version 2 to version 1 format.
    Returns the new attribute name and the converted value.
    If the value is None, None will be returned for the new value.
    """
    if value is not None:
        if attr == 'styleMapStyleName':
            value = _fontStyle2To1.get(value)
        elif attr == 'openTypeOS2WidthClass':
            value = _widthName2To1.get(value)
        elif attr == 'postscriptWindowsCharacterSet':
            value = _msCharSet2To1.get(value)
    attr = fontInfoAttributesVersion2To1.get(attr, attr)
    return (attr, value)