from __future__ import annotations
import logging
import enum
from warnings import warn
from collections import OrderedDict
import fs
import fs.base
import fs.errors
import fs.osfs
import fs.path
from fontTools.misc.textTools import tobytes
from fontTools.misc import plistlib
from fontTools.pens.pointPen import AbstractPointPen, PointToSegmentPen
from fontTools.ufoLib.errors import GlifLibError
from fontTools.ufoLib.filenames import userNameToFileName
from fontTools.ufoLib.validators import (
from fontTools.misc import etree
from fontTools.ufoLib import _UFOBaseIO, UFOFormatVersion
from fontTools.ufoLib.utils import numberTypes, _VersionTupleEnumMixin
def rebuildContents(self, validateRead=None):
    """
        Rebuild the contents dict by loading contents.plist.

        ``validateRead`` will validate the data, by default it is set to the
        class's ``validateRead`` value, can be overridden.
        """
    if validateRead is None:
        validateRead = self._validateRead
    contents = self._getPlist(CONTENTS_FILENAME, {})
    if validateRead:
        invalidFormat = False
        if not isinstance(contents, dict):
            invalidFormat = True
        else:
            for name, fileName in contents.items():
                if not isinstance(name, str):
                    invalidFormat = True
                if not isinstance(fileName, str):
                    invalidFormat = True
                elif not self.fs.exists(fileName):
                    raise GlifLibError('%s references a file that does not exist: %s' % (CONTENTS_FILENAME, fileName))
        if invalidFormat:
            raise GlifLibError('%s is not properly formatted' % CONTENTS_FILENAME)
    self.contents = contents
    self._existingFileNames = None
    self._reverseContents = None