from reportlab.platypus.flowables import Flowable, Preformatted
from reportlab import rl_config
from reportlab.lib.styles import PropertySet, ParagraphStyle, _baseFontName
from reportlab.lib import colors
from reportlab.lib.utils import annotateException, IdentStr, flatten, isStr, asNative, strTypes, __UNSET__
from reportlab.lib.validators import isListOfNumbersOrNone
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.abag import ABag as CellFrame
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus.doctemplate import Indenter, NullActionFlowable
from reportlab.platypus.flowables import LIIndenter
from collections import namedtuple
def normCellRange(self, sc, ec, sr, er):
    """ensure cell range ends are with the table bounds"""
    if sc < 0:
        sc = sc + self._ncols
    if ec < 0:
        ec = ec + self._ncols
    if sr < 0:
        sr = sr + self._nrows
    if er < 0:
        er = er + self._nrows
    return (max(0, sc), min(self._ncols - 1, ec), max(0, sr), min(self._nrows - 1, er))