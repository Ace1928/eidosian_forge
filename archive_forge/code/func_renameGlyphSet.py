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
def renameGlyphSet(self, layerName, newLayerName, defaultLayer=False):
    """
        Rename a glyph set.

        Note: if a GlyphSet object has already been retrieved for
        layerName, it is up to the caller to inform that object that
        the directory it represents has changed.
        """
    if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
        return
    if layerName == newLayerName:
        if self.layerContents[layerName] != DEFAULT_GLYPHS_DIRNAME and (not defaultLayer):
            return
        if self.layerContents[layerName] == DEFAULT_GLYPHS_DIRNAME and defaultLayer:
            return
    else:
        if newLayerName is None:
            newLayerName = DEFAULT_LAYER_NAME
        if newLayerName in self.layerContents:
            raise UFOLibError('A layer named %s already exists.' % newLayerName)
        if defaultLayer and DEFAULT_GLYPHS_DIRNAME in self.layerContents.values():
            raise UFOLibError('A default layer already exists.')
    oldDirectory = self._findDirectoryForLayerName(layerName)
    if defaultLayer:
        newDirectory = DEFAULT_GLYPHS_DIRNAME
    else:
        existing = {name.lower() for name in self.layerContents.values()}
        newDirectory = userNameToFileName(newLayerName, existing=existing, prefix='glyphs.')
    del self.layerContents[layerName]
    self.layerContents[newLayerName] = newDirectory
    self.fs.movedir(oldDirectory, newDirectory, create=True)