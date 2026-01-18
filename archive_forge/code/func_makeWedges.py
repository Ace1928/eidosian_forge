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
def makeWedges(self):
    angles = self.makeAngles()
    halfAngles = []
    for i, (a1, a2) in angles:
        if a2 is None:
            halfAngle = a1
        else:
            halfAngle = 0.5 * (a2 + a1)
        halfAngles.append(halfAngle)
    sideLabels = self.sideLabels
    n = len(angles)
    labels = _fixLabels(self.labels, n)
    wr = getattr(self, 'wedgeRecord', None)
    self._seriesCount = n
    styleCount = len(self.slices)
    plMode = self.pointerLabelMode
    if sideLabels:
        plMode = None
    if plMode:
        checkLabelOverlap = False
        PL = self.makePointerLabels(angles, plMode)
        xradius = PL.xradius
        yradius = PL.yradius
        centerx = PL.centerx
        centery = PL.centery
        PL_data = PL.data
        gSN = lambda i: ''
    else:
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
        checkLabelOverlap = self.checkLabelOverlap
        gSN = lambda i: self.getSeriesName(i, '')
    g = Group()
    g_add = g.add
    L = []
    L_add = L.append
    innerRadiusFraction = self.innerRadiusFraction
    for i, (a1, a2) in angles:
        if a2 is None:
            continue
        wedgeStyle = self.slices[i % styleCount]
        if not wedgeStyle.visible:
            continue
        aa = abs(a2 - a1)
        cx, cy = (centerx, centery)
        text = gSN(i)
        popout = wedgeStyle.popout
        if text or popout:
            averageAngle = (a1 + a2) / 2.0
            aveAngleRadians = averageAngle / _180_pi
            cosAA = cos(aveAngleRadians)
            sinAA = sin(aveAngleRadians)
            if popout and aa < _ANGLEHI:
                cx = centerx + popout * cosAA
                cy = centery + popout * sinAA
        if innerRadiusFraction:
            theWedge = Wedge(cx, cy, xradius, a1, a2, yradius=yradius, radius1=xradius * innerRadiusFraction, yradius1=yradius * innerRadiusFraction)
        elif aa >= _ANGLEHI:
            theWedge = Ellipse(cx, cy, xradius, yradius)
        else:
            theWedge = Wedge(cx, cy, xradius, a1, a2, yradius=yradius)
        theWedge.fillColor = wedgeStyle.fillColor
        theWedge.strokeColor = wedgeStyle.strokeColor
        theWedge.strokeWidth = wedgeStyle.strokeWidth
        theWedge.strokeLineJoin = wedgeStyle.strokeLineJoin
        theWedge.strokeLineCap = wedgeStyle.strokeLineCap
        theWedge.strokeMiterLimit = wedgeStyle.strokeMiterLimit
        theWedge.strokeDashArray = wedgeStyle.strokeDashArray
        shader = wedgeStyle.shadingKind
        if shader:
            nshades = aa / float(wedgeStyle.shadingAngle)
            if nshades > 1:
                shader = colors.Whiter if shader == 'lighten' else colors.Blacker
                nshades = 1 + int(nshades)
                shadingAmount = 1 - wedgeStyle.shadingAmount
                if wedgeStyle.shadingDirection == 'normal':
                    dsh = (1 - shadingAmount) / float(nshades - 1)
                    shf1 = shadingAmount
                else:
                    dsh = (shadingAmount - 1) / float(nshades - 1)
                    shf1 = 1
                shda = (a2 - a1) / float(nshades)
                shsc = wedgeStyle.fillColor
                theWedge.fillColor = None
                for ish in range(nshades):
                    sha1 = a1 + ish * shda
                    sha2 = a1 + (ish + 1) * shda
                    shc = shader(shsc, shf1 + dsh * ish)
                    if innerRadiusFraction:
                        shWedge = Wedge(cx, cy, xradius, sha1, sha2, yradius=yradius, radius1=xradius * innerRadiusFraction, yradius1=yradius * innerRadiusFraction)
                    else:
                        shWedge = Wedge(cx, cy, xradius, sha1, sha2, yradius=yradius)
                    shWedge.fillColor = shc
                    shWedge.strokeColor = None
                    shWedge.strokeWidth = 0
                    g_add(shWedge)
        g_add(theWedge)
        if wr:
            wr(theWedge, value=a1._data, label=text)
        if wedgeStyle.label_visible:
            if not sideLabels:
                if text:
                    labelRadius = wedgeStyle.labelRadius
                    rx = xradius * labelRadius
                    ry = yradius * labelRadius
                    labelX = cx + rx * cosAA
                    labelY = cy + ry * sinAA
                    l = _addWedgeLabel(self, text, averageAngle, labelX, labelY, wedgeStyle)
                    L_add(l)
                    if not plMode and l._simple_pointer:
                        l._aax = cx + xradius * cosAA
                        l._aay = cy + yradius * sinAA
                    if checkLabelOverlap:
                        l._origdata = {'x': labelX, 'y': labelY, 'angle': averageAngle, 'rx': rx, 'ry': ry, 'cx': cx, 'cy': cy, 'bounds': l.getBounds(), 'angles': (a1, a2)}
                elif plMode and PL_data:
                    l = PL_data[i]
                    if l:
                        data = l._origdata
                        sinM = data['smid']
                        cosM = data['cmid']
                        lX = cx + xradius * cosM
                        lY = cy + yradius * sinM
                        lpel = wedgeStyle.label_pointer_elbowLength
                        lXi = lX + lpel * cosM
                        lYi = lY + lpel * sinM
                        L_add(PolyLine((lX, lY, lXi, lYi, l.x, l.y), strokeWidth=wedgeStyle.label_pointer_strokeWidth, strokeColor=wedgeStyle.label_pointer_strokeColor))
                        L_add(l)
            elif text:
                slices_popout = self.slices.popout
                m = 0
                for n, angle in angles:
                    if self.slices[n].fillColor:
                        m += 1
                    else:
                        r = n % m
                        self.slices[n].fillColor = self.slices[r].fillColor
                        self.slices[n].popout = self.slices[r].popout
                for j in range(0, m - 1):
                    if self.slices[j].popout > slices_popout:
                        slices_popout = self.slices[j].popout
                labelRadius = wedgeStyle.labelRadius
                ry = yradius * labelRadius
                if abs(averageAngle) < 90 or (averageAngle > 270 and averageAngle < 450) or -450 < averageAngle < -270:
                    labelX = (1 + self.sideLabelsOffset) * self.width + self.x + slices_popout
                    rx = 0
                else:
                    labelX = self.x - self.sideLabelsOffset * self.width - slices_popout
                    rx = 0
                labelY = cy + ry * sinAA
                l = _addWedgeLabel(self, text, averageAngle, labelX, labelY, wedgeStyle)
                L_add(l)
                if not plMode:
                    l._aax = cx + xradius * cosAA
                    l._aay = cy + yradius * sinAA
                if checkLabelOverlap:
                    l._origdata = {'x': labelX, 'y': labelY, 'angle': averageAngle, 'rx': rx, 'ry': ry, 'cx': cx, 'cy': cy, 'bounds': l.getBounds()}
                x1, y1, x2, y2 = l.getBounds()
    if checkLabelOverlap and L:
        fixLabelOverlaps(L, sideLabels, mult0=checkLabelOverlap)
    for l in L:
        g_add(l)
    if not plMode:
        for l in L:
            if l._simple_pointer and (not sideLabels):
                g_add(Line(l.x, l.y, l._aax, l._aay, strokeWidth=wedgeStyle.label_pointer_strokeWidth, strokeColor=wedgeStyle.label_pointer_strokeColor))
            elif sideLabels:
                x1, y1, x2, y2 = l.getBounds()
                if l.x == (1 + self.sideLabelsOffset) * self.width + self.x:
                    g_add(Line(l._aax, l._aay, 0.5 * (l._aax + l.x), l.y + 0.25 * (y2 - y1), strokeWidth=wedgeStyle.label_pointer_strokeWidth, strokeColor=wedgeStyle.label_pointer_strokeColor))
                    g_add(Line(0.5 * (l._aax + l.x), l.y + 0.25 * (y2 - y1), l.x, l.y + 0.25 * (y2 - y1), strokeWidth=wedgeStyle.label_pointer_strokeWidth, strokeColor=wedgeStyle.label_pointer_strokeColor))
                else:
                    g_add(Line(l._aax, l._aay, 0.5 * (l._aax + l.x), l.y + 0.25 * (y2 - y1), strokeWidth=wedgeStyle.label_pointer_strokeWidth, strokeColor=wedgeStyle.label_pointer_strokeColor))
                    g_add(Line(0.5 * (l._aax + l.x), l.y + 0.25 * (y2 - y1), l.x, l.y + 0.25 * (y2 - y1), strokeWidth=wedgeStyle.label_pointer_strokeWidth, strokeColor=wedgeStyle.label_pointer_strokeColor))
    return g