from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
def glyphSet(self):
    """The glyphs in this class as a tuple of :class:`GlyphName` objects."""
    return self.glyphs.glyphSet()