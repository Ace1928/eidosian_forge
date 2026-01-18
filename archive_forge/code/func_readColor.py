from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def readColor(text):
    """Read color names or tuples, RGB or CMYK, and return a Color object."""
    if not text:
        return None
    from reportlab.lib import colors
    from string import letters
    if text[0] in letters:
        return colors.__dict__[text]
    tup = lengthSequence(text)
    msg = 'Color tuple must have 3 (or 4) elements for RGB (or CMYC).'
    assert 3 <= len(tup) <= 4, msg
    msg = 'Color tuple must have all elements <= 1.0.'
    for i in range(len(tup)):
        assert tup[i] <= 1.0, msg
    if len(tup) == 3:
        colClass = colors.Color
    elif len(tup) == 4:
        colClass = colors.CMYKColor
    return colClass(*tup)