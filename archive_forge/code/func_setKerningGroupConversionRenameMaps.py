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
def setKerningGroupConversionRenameMaps(self, maps):
    """
        Set maps defining the renaming that should be done
        when writing groups and kerning in UFO 1 and UFO 2.
        This will effectively undo the conversion done when
        UFOReader reads this data. The dictionary should have
        this form::

                {
                        "side1" : {"group name to use when writing" : "group name in data"},
                        "side2" : {"group name to use when writing" : "group name in data"}
                }

        This is the same form returned by UFOReader's
        getKerningGroupConversionRenameMaps method.
        """
    if self._formatVersion >= UFOFormatVersion.FORMAT_3_0:
        return
    remap = {}
    for side in ('side1', 'side2'):
        for writeName, dataName in list(maps[side].items()):
            remap[dataName] = writeName
    self._downConversionKerningData = dict(groupRenameMap=remap)