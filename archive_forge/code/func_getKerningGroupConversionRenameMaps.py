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
def getKerningGroupConversionRenameMaps(self, validate=None):
    """
        Get maps defining the renaming that was done during any
        needed kerning group conversion. This method returns a
        dictionary of this form::

                {
                        "side1" : {"old group name" : "new group name"},
                        "side2" : {"old group name" : "new group name"}
                }

        When no conversion has been performed, the side1 and side2
        dictionaries will be empty.

        ``validate`` will validate the groups, by default it is set to the
        class's validate value, can be overridden.
        """
    if validate is None:
        validate = self._validate
    if self._formatVersion >= UFOFormatVersion.FORMAT_3_0:
        return dict(side1={}, side2={})
    self.readGroups(validate=validate)
    return self._upConvertedKerningData['groupRenameMaps']