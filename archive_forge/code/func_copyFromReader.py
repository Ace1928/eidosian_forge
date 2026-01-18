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
def copyFromReader(self, reader, sourcePath, destPath):
    """
        Copy the sourcePath in the provided UFOReader to destPath
        in this writer. The paths must be relative. This works with
        both individual files and directories.
        """
    if not isinstance(reader, UFOReader):
        raise UFOLibError('The reader must be an instance of UFOReader.')
    sourcePath = fsdecode(sourcePath)
    destPath = fsdecode(destPath)
    if not reader.fs.exists(sourcePath):
        raise UFOLibError('The reader does not have data located at "%s".' % sourcePath)
    if self.fs.exists(destPath):
        raise UFOLibError('A file named "%s" already exists.' % destPath)
    self.fs.makedirs(fs.path.dirname(destPath), recreate=True)
    if reader.fs.isdir(sourcePath):
        fs.copy.copy_dir(reader.fs, sourcePath, self.fs, destPath)
    else:
        fs.copy.copy_file(reader.fs, sourcePath, self.fs, destPath)