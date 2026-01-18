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
def getComponentReferences(self, glyphNames=None):
    """
        Return a dictionary that maps glyph names to lists containing the
        base glyph name of components in the glyph. This parses the .glif
        files partially, so it is a lot faster than parsing all files completely.
        By default this checks all glyphs, but a subset can be passed with glyphNames.
        """
    components = {}
    if glyphNames is None:
        glyphNames = self.contents.keys()
    for glyphName in glyphNames:
        text = self.getGLIF(glyphName)
        components[glyphName] = _fetchComponentBases(text)
    return components