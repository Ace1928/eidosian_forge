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
def makeTicks(self):
    xold = self._x
    try:
        self._x = self._labelAxisPos(getattr(self, 'tickAxisMode', 'axis'))
        g = self._drawTicks(self.tickRight, self.tickLeft)
        self._drawSubTicks(getattr(self, 'subTickHi', 0), getattr(self, 'subTickLo', 0), g)
        return g
    finally:
        self._x = xold