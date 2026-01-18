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
def readGlyphFromString(aString, glyphObject=None, pointPen=None, formatVersions=None, validate=True):
    """
    Read .glif data from a string into a glyph object.

    The 'glyphObject' argument can be any kind of object (even None);
    the readGlyphFromString() method will attempt to set the following
    attributes on it:

    width
            the advance width of the glyph
    height
            the advance height of the glyph
    unicodes
            a list of unicode values for this glyph
    note
            a string
    lib
            a dictionary containing custom data
    image
            a dictionary containing image data
    guidelines
            a list of guideline data dictionaries
    anchors
            a list of anchor data dictionaries

    All attributes are optional, in two ways:

    1) An attribute *won't* be set if the .glif file doesn't
       contain data for it. 'glyphObject' will have to deal
       with default values itself.
    2) If setting the attribute fails with an AttributeError
       (for example if the 'glyphObject' attribute is read-
       only), readGlyphFromString() will not propagate that
       exception, but ignore that attribute.

    To retrieve outline information, you need to pass an object
    conforming to the PointPen protocol as the 'pointPen' argument.
    This argument may be None if you don't need the outline data.

    The formatVersions optional argument define the GLIF format versions
    that are allowed to be read.
    The type is Optional[Iterable[Tuple[int, int], int]]. It can contain
    either integers (for the major versions to be allowed, with minor
    digits defaulting to 0), or tuples of integers to specify both
    (major, minor) versions.
    By default when formatVersions is None all the GLIF format versions
    currently defined are allowed to be read.

    ``validate`` will validate the read data. It is set to ``True`` by default.
    """
    tree = _glifTreeFromString(aString)
    if formatVersions is None:
        validFormatVersions = GLIFFormatVersion.supported_versions()
    else:
        validFormatVersions, invalidFormatVersions = (set(), set())
        for v in formatVersions:
            try:
                formatVersion = GLIFFormatVersion(v)
            except ValueError:
                invalidFormatVersions.add(v)
            else:
                validFormatVersions.add(formatVersion)
        if not validFormatVersions:
            raise ValueError(f'None of the requested GLIF formatVersions are supported: {formatVersions!r}')
    _readGlyphFromTree(tree, glyphObject, pointPen, formatVersions=validFormatVersions, validate=validate)