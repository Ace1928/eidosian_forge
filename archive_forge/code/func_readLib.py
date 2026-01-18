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
def readLib(self, validate=None):
    """
        Read lib.plist. Returns a dict.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    if validate is None:
        validate = self._validate
    data = self._getPlist(LIB_FILENAME, {})
    if validate:
        valid, message = fontLibValidator(data)
        if not valid:
            raise UFOLibError(message)
    return data