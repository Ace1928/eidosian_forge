from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ
from reportlab.lib.utils import isUnicode
import re
def kinsokuShoriSplit(word, widths, availWidth):
    """Split according to Japanese rules according to CJKV (Lunde).

    Essentially look for "nice splits" so that we don't end a line
    with an open bracket, or start one with a full stop, or stuff like
    that.  There is no attempt to try to split compound words into
    constituent kanji.  It currently uses wrap-down: packs as much
    on a line as possible, then backtracks if needed

    This returns a number of words each of which should just about fit
    on a line.  If you give it a whole paragraph at once, it will
    do all the splits.

    It's possible we might slightly step over the width limit
    if we do hanging punctuation marks in future (e.g. dangle a Japanese
    full stop in the right margin rather than using a whole character
    box.

    """
    lines = []
    assert len(word) == len(widths)
    curWidth = 0.0
    curLine = []
    i = 0
    while 1:
        ch = word[i]
        w = widths[i]
        if curWidth + w < availWidth:
            curLine.append(ch)
            curWidth += w
        elif ch in CANNOT_END_LINE[0]:
            pass