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
def makeGrid(self, g, dim=None, parent=None, exclude=[]):
    """this is only called by a container object"""
    c = self.gridStrokeColor
    w = self.gridStrokeWidth or 0
    if w and c and self.visibleGrid:
        s = self.gridStart
        e = self.gridEnd
        if s is None or e is None:
            if dim and hasattr(dim, '__call__'):
                dim = dim()
            if dim:
                if s is None:
                    s = dim[0]
                if e is None:
                    e = dim[1]
            else:
                if s is None:
                    s = 0
                if e is None:
                    e = 0
        if s or e:
            if self.isYAxis:
                offs = self._x
            else:
                offs = self._y
            self._makeLines(g, s - offs, e - offs, c, w, self.gridStrokeDashArray, self.gridStrokeLineJoin, self.gridStrokeLineCap, self.gridStrokeMiterLimit, parent=parent, exclude=exclude, specials=getattr(self, '_gridSpecials', {}))
    self._makeSubGrid(g, dim, parent, exclude=[])