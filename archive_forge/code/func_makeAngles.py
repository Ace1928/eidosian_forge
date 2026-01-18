import functools
from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfNumbersOrNone,\
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Ellipse, Wedge, String, STATE_DEFAULTS, ArcPath, Polygon, Rect, PolyLine, Line
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab.graphics.charts.textlabels import Label
from reportlab import cmp
from reportlab.graphics.charts.utils3d import _getShaded, _2rad, _360, _180_pi
def makeAngles(self):
    wr = getattr(self, 'wedgeRecord', None)
    if self.sideLabels:
        startAngle = theta0(self.data, self.direction)
        self.slices.label_visible = 1
    else:
        startAngle = self.startAngle % 360
    whichWay = self.direction == 'clockwise' and -1 or 1
    D = [a for a in enumerate(self.normalizeData(keepData=wr))]
    if self.orderMode == 'alternate' and (not self.sideLabels):
        W = [a for a in D if abs(a[1]) >= 1e-05]
        W.sort(key=_arcCF)
        T = [[], []]
        i = 0
        while W:
            if i < 2:
                a = W.pop(0)
            else:
                a = W.pop(-1)
            T[i % 2].append(a)
            i += 1
            i %= 4
        T[1].reverse()
        D = T[0] + T[1] + [a for a in D if abs(a[1]) < 1e-05]
    A = []
    a = A.append
    for i, angle in D:
        endAngle = startAngle + angle * whichWay
        if abs(angle) >= _ANGLELO:
            if startAngle >= endAngle:
                aa = (endAngle, startAngle)
            else:
                aa = (startAngle, endAngle)
        else:
            aa = (startAngle, None)
        if wr:
            aa = (AngleData(aa[0], angle._data), aa[1])
        startAngle = endAngle
        a((i, aa))
    return A