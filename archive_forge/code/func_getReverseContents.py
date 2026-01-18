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
def getReverseContents(self):
    """
        Return a reversed dict of self.contents, mapping file names to
        glyph names. This is primarily an aid for custom glyph name to file
        name schemes that want to make sure they don't generate duplicate
        file names. The file names are converted to lowercase so we can
        reliably check for duplicates that only differ in case, which is
        important for case-insensitive file systems.
        """
    if self._reverseContents is None:
        d = {}
        for k, v in self.contents.items():
            d[v.lower()] = k
        self._reverseContents = d
    return self._reverseContents