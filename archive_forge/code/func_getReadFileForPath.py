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
def getReadFileForPath(self, path, encoding=None):
    """
        Returns a file (or file-like) object for the file at the given path.
        The path must be relative to the UFO path.
        Returns None if the file does not exist.
        By default the file is opened in binary mode (reads bytes).
        If encoding is passed, the file is opened in text mode (reads str).

        Note: The caller is responsible for closing the open file.
        """
    path = fsdecode(path)
    try:
        if encoding is None:
            return self.fs.openbin(path)
        else:
            return self.fs.open(path, mode='r', encoding=encoding)
    except fs.errors.ResourceNotFound:
        return None