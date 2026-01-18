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
def makePointerLabels(self, angles, plMode):

    class PL:

        def __init__(self, centerx, centery, xradius, yradius, data, lu=0, ru=0):
            self.centerx = centerx
            self.centery = centery
            self.xradius = xradius
            self.yradius = yradius
            self.data = data
            self.lu = lu
            self.ru = ru
    labelX = self.width - 2
    labelY = self.height
    n = nr = nl = maxW = sumH = 0
    styleCount = len(self.slices)
    L = []
    L_add = L.append
    refArcs = _makeSideArcDefs(self.startAngle, self.direction)
    for i, A in angles:
        if A[1] is None:
            continue
        sn = self.getSeriesName(i, '')
        if not sn:
            continue
        style = self.slices[i % styleCount]
        if not style.label_visible or not style.visible:
            continue
        n += 1
        l = _addWedgeLabel(self, sn, 180, labelX, labelY, style)
        L_add(l)
        b = l.getBounds()
        w = b[2] - b[0]
        h = b[3] - b[1]
        ri = [(a[0], intervalIntersection(A, (a[1], a[2]))) for a in refArcs]
        li = _findLargestArc(ri, 0)
        ri = _findLargestArc(ri, 1)
        if li and ri:
            if plMode == 'LeftAndRight':
                if li[1] - li[0] < ri[1] - ri[0]:
                    li = None
                else:
                    ri = None
            elif li[1] - li[0] < 0.02 * (ri[1] - ri[0]):
                li = None
            elif (li[1] - li[0]) * 0.02 > ri[1] - ri[0]:
                ri = None
        if ri:
            nr += 1
        if li:
            nl += 1
        l._origdata = dict(bounds=b, width=w, height=h, li=li, ri=ri, index=i, edgePad=style.label_pointer_edgePad, piePad=style.label_pointer_piePad, elbowLength=style.label_pointer_elbowLength)
        maxW = max(w, maxW)
        sumH += h + 2
    if not n:
        xradius = self.width * 0.5
        yradius = self.height * 0.5
        centerx = self.x + xradius
        centery = self.y + yradius
        if self.xradius:
            xradius = self.xradius
        if self.yradius:
            yradius = self.yradius
        if self.sameRadii:
            xradius = yradius = min(xradius, yradius)
        return PL(centerx, centery, xradius, yradius, [])
    aonR = nr == n
    if sumH < self.height and (aonR or nl == n):
        side = int(aonR)
    else:
        side = None
    G, lu, ru, mel = _fixPointerLabels(len(angles), L, self.x, self.y, self.width, self.height, side=side)
    if plMode == 'LeftAndRight':
        lu = ru = max(lu, ru)
    x0 = self.x + lu
    x1 = self.x + self.width - ru
    xradius = (x1 - x0) * 0.5
    yradius = self.height * 0.5 - mel
    centerx = x0 + xradius
    centery = self.y + yradius + mel
    if self.xradius:
        xradius = self.xradius
    if self.yradius:
        yradius = self.yradius
    if self.sameRadii:
        xradius = yradius = min(xradius, yradius)
    return PL(centerx, centery, xradius, yradius, G, lu, ru)