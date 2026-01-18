from fontTools.config import Config
from fontTools.misc import xmlWriter
from fontTools.misc.configTools import AbstractConfig
from fontTools.misc.textTools import Tag, byteord, tostr
from fontTools.misc.loggingTools import deprecateArgument
from fontTools.ttLib import TTLibError
from fontTools.ttLib.ttGlyphSet import _TTGlyph, _TTGlyphSetCFF, _TTGlyphSetGlyf
from fontTools.ttLib.sfnt import SFNTReader, SFNTWriter
from io import BytesIO, StringIO, UnsupportedOperation
import os
import logging
import traceback
def normalizeLocation(self, location):
    """Normalize a ``location`` from the font's defined axes space (also
        known as user space) into the normalized (-1..+1) space. It applies
        ``avar`` mapping if the font contains an ``avar`` table.

        The ``location`` parameter should be a dictionary mapping four-letter
        variation tags to their float values.

        Raises ``TTLibError`` if the font is not a variable font.
        """
    from fontTools.varLib.models import normalizeLocation, piecewiseLinearMap
    if 'fvar' not in self:
        raise TTLibError('Not a variable font')
    axes = {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in self['fvar'].axes}
    location = normalizeLocation(location, axes)
    if 'avar' in self:
        avar = self['avar']
        avarSegments = avar.segments
        mappedLocation = {}
        for axisTag, value in location.items():
            avarMapping = avarSegments.get(axisTag, None)
            if avarMapping is not None:
                value = piecewiseLinearMap(value, avarMapping)
            mappedLocation[axisTag] = value
        location = mappedLocation
    return location