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
def setModificationTime(self):
    """
        Set the UFO modification time to the current time.
        This is never called automatically. It is up to the
        caller to call this when finished working on the UFO.
        """
    path = self._path
    if path is not None and os.path.exists(path):
        try:
            os.utime(path, None)
        except OSError as e:
            logger.warning('Failed to set modified time: %s', e)