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
def readImage(self, fileName, validate=None):
    """
        Return image data for the file named fileName.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
    if validate is None:
        validate = self._validate
    if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
        raise UFOLibError(f'Reading images is not allowed in UFO {self._formatVersion.major}.')
    fileName = fsdecode(fileName)
    try:
        try:
            imagesFS = self._imagesFS
        except AttributeError:
            imagesFS = self.fs.opendir(IMAGES_DIRNAME)
        data = imagesFS.readbytes(fileName)
    except fs.errors.ResourceNotFound:
        raise UFOLibError(f"No image file named '{fileName}' on {self.fs}")
    if validate:
        valid, error = pngValidator(data=data)
        if not valid:
            raise UFOLibError(error)
    return data