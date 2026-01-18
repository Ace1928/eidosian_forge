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
def makeAxis(self):
    g = Group()
    self._joinToAxis()
    if not self.visibleAxis:
        return g
    axis = Line(self._x, self._y - self.loLLen, self._x, self._y + self._length + self.hiLLen)
    axis.strokeColor = self.strokeColor
    axis.strokeWidth = self.strokeWidth
    axis.strokeDashArray = self.strokeDashArray
    g.add(axis)
    return g