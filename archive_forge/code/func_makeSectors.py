from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfStringsOrNone, OneOf,\
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Wedge
from reportlab.graphics.widgetbase import TypedPropertyCollection
from reportlab.graphics.charts.piecharts import AbstractPieChart, WedgeProperties, _addWedgeLabel, fixLabelOverlaps
from functools import reduce
def makeSectors(self):
    data = self.data
    multi = isListOfListOfNoneOrNumber(data)
    if multi:
        normData = []
        n = []
        for l in data:
            t = self.normalizeData(l)
            normData.append(t)
            n.append(len(t))
        self._seriesCount = max(n)
    else:
        normData = self.normalizeData(data)
        n = len(normData)
        self._seriesCount = n
    checkLabelOverlap = self.checkLabelOverlap
    L = []
    L_add = L.append
    labels = self.labels
    if labels is None:
        labels = []
        if not multi:
            labels = [''] * n
        else:
            for m in n:
                labels = list(labels) + [''] * m
    elif not multi:
        i = n - len(labels)
        if i > 0:
            labels = list(labels) + [''] * i
    else:
        tlab = 0
        for m in n:
            tlab += m
        i = tlab - len(labels)
        if i > 0:
            labels = list(labels) + [''] * i
    self.labels = labels
    xradius = self.width / 2.0
    yradius = self.height / 2.0
    centerx = self.x + xradius
    centery = self.y + yradius
    if self.direction == 'anticlockwise':
        whichWay = 1
    else:
        whichWay = -1
    g = Group()
    startAngle = self.startAngle
    styleCount = len(self.slices)
    irf = self.innerRadiusFraction
    if multi:
        ndata = len(data)
        if irf is None:
            yir = yradius / 2.5 / ndata
            xir = xradius / 2.5 / ndata
        else:
            yir = yradius * irf
            xir = xradius * irf
        ydr = (yradius - yir) / ndata
        xdr = (xradius - xir) / ndata
        for sn, series in enumerate(normData):
            for i, angle in enumerate(series):
                endAngle = startAngle + angle * whichWay
                aa = abs(startAngle - endAngle)
                if aa < 1e-05:
                    startAngle = endAngle
                    continue
                if startAngle < endAngle:
                    a1 = startAngle
                    a2 = endAngle
                else:
                    a1 = endAngle
                    a2 = startAngle
                startAngle = endAngle
                sectorStyle = self.slices[sn, i % styleCount]
                cx, cy = (centerx, centery)
                if sectorStyle.popout != 0:
                    averageAngle = (a1 + a2) / 2.0
                    aveAngleRadians = averageAngle * pi / 180.0
                    popdistance = sectorStyle.popout
                    cx = centerx + popdistance * cos(aveAngleRadians)
                    cy = centery + popdistance * sin(aveAngleRadians)
                yr1 = yir + sn * ydr
                yr = yr1 + ydr
                xr1 = xir + sn * xdr
                xr = xr1 + xdr
                if len(series) > 1:
                    theSector = Wedge(cx, cy, xr, a1, a2, yradius=yr, radius1=xr1, yradius1=yr1)
                else:
                    theSector = Wedge(cx, cy, xr, a1, a2, yradius=yr, radius1=xr1, yradius1=yr1, annular=True)
                theSector.fillColor = sectorStyle.fillColor
                theSector.strokeColor = sectorStyle.strokeColor
                theSector.strokeWidth = sectorStyle.strokeWidth
                theSector.strokeDashArray = sectorStyle.strokeDashArray
                shader = sectorStyle.shadingKind
                if shader:
                    nshades = aa / float(sectorStyle.shadingAngle)
                    if nshades > 1:
                        shader = colors.Whiter if shader == 'lighten' else colors.Blacker
                        nshades = 1 + int(nshades)
                        shadingAmount = 1 - sectorStyle.shadingAmount
                        if sectorStyle.shadingDirection == 'normal':
                            dsh = (1 - shadingAmount) / float(nshades - 1)
                            shf1 = shadingAmount
                        else:
                            dsh = (shadingAmount - 1) / float(nshades - 1)
                            shf1 = 1
                        shda = (a2 - a1) / float(nshades)
                        shsc = sectorStyle.fillColor
                        theSector.fillColor = None
                        for ish in range(nshades):
                            sha1 = a1 + ish * shda
                            sha2 = a1 + (ish + 1) * shda
                            shc = shader(shsc, shf1 + dsh * ish)
                            if len(series) > 1:
                                shSector = Wedge(cx, cy, xr, sha1, sha2, yradius=yr, radius1=xr1, yradius1=yr1)
                            else:
                                shSector = Wedge(cx, cy, xr, sha1, sha2, yradius=yr, radius1=xr1, yradius1=yr1, annular=True)
                            shSector.fillColor = shc
                            shSector.strokeColor = None
                            shSector.strokeWidth = 0
                            g.add(shSector)
                g.add(theSector)
                if sn == 0 and sectorStyle.visible and sectorStyle.label_visible:
                    text = self.getSeriesName(i, '')
                    if text:
                        averageAngle = (a1 + a2) / 2.0
                        aveAngleRadians = averageAngle * pi / 180.0
                        labelRadius = sectorStyle.labelRadius
                        rx = xradius * labelRadius
                        ry = yradius * labelRadius
                        labelX = centerx + 0.5 * self.width * cos(aveAngleRadians) * labelRadius
                        labelY = centery + 0.5 * self.height * sin(aveAngleRadians) * labelRadius
                        l = _addWedgeLabel(self, text, averageAngle, labelX, labelY, sectorStyle)
                        if checkLabelOverlap:
                            l._origdata = {'x': labelX, 'y': labelY, 'angle': averageAngle, 'rx': rx, 'ry': ry, 'cx': cx, 'cy': cy, 'bounds': l.getBounds()}
                        L_add(l)
    else:
        if irf is None:
            yir = yradius / 2.5
            xir = xradius / 2.5
        else:
            yir = yradius * irf
            xir = xradius * irf
        for i, angle in enumerate(normData):
            endAngle = startAngle + angle * whichWay
            aa = abs(startAngle - endAngle)
            if aa < 1e-05:
                startAngle = endAngle
                continue
            if startAngle < endAngle:
                a1 = startAngle
                a2 = endAngle
            else:
                a1 = endAngle
                a2 = startAngle
            startAngle = endAngle
            sectorStyle = self.slices[i % styleCount]
            cx, cy = (centerx, centery)
            if sectorStyle.popout != 0:
                averageAngle = (a1 + a2) / 2.0
                aveAngleRadians = averageAngle * pi / 180.0
                popdistance = sectorStyle.popout
                cx = centerx + popdistance * cos(aveAngleRadians)
                cy = centery + popdistance * sin(aveAngleRadians)
            if n > 1:
                theSector = Wedge(cx, cy, xradius, a1, a2, yradius=yradius, radius1=xir, yradius1=yir)
            elif n == 1:
                theSector = Wedge(cx, cy, xradius, a1, a2, yradius=yradius, radius1=xir, yradius1=yir, annular=True)
            theSector.fillColor = sectorStyle.fillColor
            theSector.strokeColor = sectorStyle.strokeColor
            theSector.strokeWidth = sectorStyle.strokeWidth
            theSector.strokeDashArray = sectorStyle.strokeDashArray
            shader = sectorStyle.shadingKind
            if shader:
                nshades = aa / float(sectorStyle.shadingAngle)
                if nshades > 1:
                    shader = colors.Whiter if shader == 'lighten' else colors.Blacker
                    nshades = 1 + int(nshades)
                    shadingAmount = 1 - sectorStyle.shadingAmount
                    if sectorStyle.shadingDirection == 'normal':
                        dsh = (1 - shadingAmount) / float(nshades - 1)
                        shf1 = shadingAmount
                    else:
                        dsh = (shadingAmount - 1) / float(nshades - 1)
                        shf1 = 1
                    shda = (a2 - a1) / float(nshades)
                    shsc = sectorStyle.fillColor
                    theSector.fillColor = None
                    for ish in range(nshades):
                        sha1 = a1 + ish * shda
                        sha2 = a1 + (ish + 1) * shda
                        shc = shader(shsc, shf1 + dsh * ish)
                        if n > 1:
                            shSector = Wedge(cx, cy, xradius, sha1, sha2, yradius=yradius, radius1=xir, yradius1=yir)
                        elif n == 1:
                            shSector = Wedge(cx, cy, xradius, sha1, sha2, yradius=yradius, radius1=xir, yradius1=yir, annular=True)
                        shSector.fillColor = shc
                        shSector.strokeColor = None
                        shSector.strokeWidth = 0
                        g.add(shSector)
            g.add(theSector)
            if labels[i] and sectorStyle.visible and sectorStyle.label_visible:
                averageAngle = (a1 + a2) / 2.0
                aveAngleRadians = averageAngle * pi / 180.0
                labelRadius = sectorStyle.labelRadius
                labelX = centerx + 0.5 * self.width * cos(aveAngleRadians) * labelRadius
                labelY = centery + 0.5 * self.height * sin(aveAngleRadians) * labelRadius
                rx = xradius * labelRadius
                ry = yradius * labelRadius
                l = _addWedgeLabel(self, labels[i], averageAngle, labelX, labelY, sectorStyle)
                if checkLabelOverlap:
                    l._origdata = {'x': labelX, 'y': labelY, 'angle': averageAngle, 'rx': rx, 'ry': ry, 'cx': cx, 'cy': cy, 'bounds': l.getBounds()}
                L_add(l)
    if checkLabelOverlap and L:
        fixLabelOverlaps(L)
    for l in L:
        g.add(l)
    return g