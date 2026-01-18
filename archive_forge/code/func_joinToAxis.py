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
def joinToAxis(self, xAxis, mode='left', pos=None):
    """Join with x-axis using some mode."""
    _assertXAxis(xAxis)
    if mode == 'left':
        self._x = xAxis._x * 1.0
    elif mode == 'right':
        self._x = (xAxis._x + xAxis._length) * 1.0
    elif mode == 'value':
        self._x = xAxis.scale(pos) * 1.0
    elif mode == 'points':
        self._x = pos * 1.0