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
def getFileObjectForPath(self, path, mode='w', encoding=None):
    """
        Returns a file (or file-like) object for the
        file at the given path. The path must be relative
        to the UFO path. Returns None if the file does
        not exist and the mode is "r" or "rb.
        An encoding may be passed if the file is opened in text mode.

        Note: The caller is responsible for closing the open file.
        """
    path = fsdecode(path)
    try:
        return self.fs.open(path, mode=mode, encoding=encoding)
    except fs.errors.ResourceNotFound as e:
        m = mode[0]
        if m == 'r':
            return None
        elif m == 'w' or m == 'a' or m == 'x':
            self.fs.makedirs(fs.path.dirname(path), recreate=True)
            return self.fs.open(path, mode=mode, encoding=encoding)
    except fs.errors.ResourceError as e:
        return UFOLibError(f"unable to open '{path}' on {self.fs}: {e}")