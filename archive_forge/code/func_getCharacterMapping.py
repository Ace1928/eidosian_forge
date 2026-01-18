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
def getCharacterMapping(self, layerName=None, validate=None):
    """
        Return a dictionary that maps unicode values (ints) to
        lists of glyph names.
        """
    if validate is None:
        validate = self._validate
    glyphSet = self.getGlyphSet(layerName, validateRead=validate, validateWrite=True)
    allUnicodes = glyphSet.getUnicodes()
    cmap = {}
    for glyphName, unicodes in allUnicodes.items():
        for code in unicodes:
            if code in cmap:
                cmap[code].append(glyphName)
            else:
                cmap[code] = [glyphName]
    return cmap