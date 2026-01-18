from math import log10 as math_log10
from reportlab.lib.validators import    isNumber, isNumberOrNone, isListOfStringsOrNone, isListOfNumbers, \
from reportlab.lib.attrmap import *
from reportlab.lib import normalDate
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, Group, STATE_DEFAULTS, _textBoxLimits, _rotatedBoxLimits
from reportlab.graphics.widgetbase import Widget, TypedPropertyCollection
from reportlab.graphics.charts.textlabels import Label, PMVLabel, XLabel,  DirectDrawFlowable
from reportlab.graphics.charts.utils import nextRoundNumber
from reportlab.graphics.widgets.grids import ShadedRect
from reportlab.lib.colors import Color
from reportlab.lib.utils import isSeq
def parseDayAndMonth(dmstr):
    """This accepts and validates strings like "31-Dec" i.e. dates
    of no particular year.  29 Feb is allowed.  These can be used
    for recurring dates.  It returns a (dd, mm) pair where mm is the
    month integer.  If the text is not valid it raises an error.
    """
    dstr, mstr = dmstr.split('-')
    dd = int(dstr)
    mstr = mstr.lower()
    mm = _months.index(mstr) + 1
    assert dd <= _maxDays[mm - 1]
    return (dd, mm)