from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
def splitString(self, text, doc, encoding='utf-8'):
    """Splits text into a number of chunks, each of which belongs to a
        single subset.  Returns a list of tuples (subset, string).  Use subset
        numbers with getSubsetInternalName.  Doc is needed for distinguishing
        subsets when building different documents at the same time."""
    asciiReadable = self._asciiReadable
    try:
        state = self.state[doc]
    except KeyError:
        state = self.state[doc] = TTFont.State(asciiReadable, self)
    _31skip = 31 if asciiReadable and state.nextCode < 32 else -256
    curSet = -1
    cur = []
    results = []
    if not isUnicode(text):
        text = text.decode('utf-8')
    charToGlyph = self.face.charToGlyph
    assignments = state.assignments
    subsets = state.subsets
    for code in map(ord, text):
        if code == 160:
            code = 32
        if code in assignments:
            n = assignments[code]
        elif code not in charToGlyph:
            n = 0
        else:
            if state.frozen:
                raise pdfdoc.PDFError('Font %s is already frozen, cannot add new character U+%04X' % (self.fontName, code))
            n = state.nextCode
            if n & 255 == 32:
                if n != 32:
                    subsets[n >> 8].append(32)
                state.nextCode += 1
                n = state.nextCode
            if n > 32:
                if not n & 255:
                    subsets.append([0])
                    state.nextCode += 1
                    n = state.nextCode
                subsets[n >> 8].append(code)
            else:
                if n == _31skip:
                    state.nextCode = 127
                subsets[0][n] = code
            state.nextCode += 1
            assignments[code] = n
        if n >> 8 != curSet:
            if cur:
                results.append((curSet, bytes(cur)))
            curSet = n >> 8
            cur = []
        cur.append(n & 255)
    if cur:
        results.append((curSet, bytes(cur)))
    return results