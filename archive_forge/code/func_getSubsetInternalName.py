from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def getSubsetInternalName(self, subset, doc):
    """Returns the name of a PDF Font object corresponding to a given
        subset of this dynamic font.  Use this function instead of
        PDFDocument.getInternalFontName."""
    try:
        state = self.state[doc]
    except KeyError:
        state = self.state[doc] = TTFont.State(self._asciiReadable)
    if subset < 0 or subset >= len(state.subsets):
        raise IndexError('Subset %d does not exist in font %s' % (subset, self.fontName))
    if state.internalName is None:
        state.internalName = state.namePrefix + repr(len(doc.fontMapping) + 1)
        doc.fontMapping[self.fontName] = '/' + state.internalName
        doc.delayedFonts.append(self)
    return '/%s+%d' % (state.internalName, subset)