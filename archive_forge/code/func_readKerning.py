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
def readKerning(self, validate=None):
    """
        Read kerning.plist. Returns a dict.

        ``validate`` will validate the kerning data, by default it is set to the
        class's validate value, can be overridden.
        """
    if validate is None:
        validate = self._validate
    if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
        self._upConvertKerning(validate)
        kerningNested = self._upConvertedKerningData['kerning']
    else:
        kerningNested = self._readKerning()
    if validate:
        valid, message = kerningValidator(kerningNested)
        if not valid:
            raise UFOLibError(message)
    kerning = {}
    for left in kerningNested:
        for right in kerningNested[left]:
            value = kerningNested[left][right]
            kerning[left, right] = value
    return kerning