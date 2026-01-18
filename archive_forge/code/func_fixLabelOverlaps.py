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
def fixLabelOverlaps(L, sideLabels=False, mult0=1.0):
    nL = len(L)
    if nL < 2:
        return
    B = [l._origdata['bounds'] for l in L]
    OK = 1
    RP = []
    iter = 0
    mult0 = float(mult0 + 0)
    mult = mult0
    if not sideLabels:
        while iter < 30:
            R = findOverlapRun(B)
            if not R:
                break
            nR = len(R)
            if nR == nL:
                break
            if not [r for r in RP if r in R]:
                mult = mult0
            da = 0
            r0 = R[0]
            rL = R[-1]
            bi = B[r0]
            taa = aa = _360(L[r0]._pmv)
            for r in R[1:]:
                b = B[r]
                da = max(da, min(b[2] - bi[0], bi[2] - b[0]))
                bi = b
                aa += L[r]._pmv
            aa = aa / float(nR)
            utaa = abs(L[rL]._pmv - taa)
            ntaa = _360(utaa)
            da *= mult * (nR - 1) / ntaa
            for r in R:
                l = L[r]
                orig = l._origdata
                angle = l._pmv = _360(l._pmv + da * (_360(l._pmv) - aa))
                rad = angle / _180_pi
                l.x = orig['cx'] + orig['rx'] * cos(rad)
                l.y = orig['cy'] + orig['ry'] * sin(rad)
                B[r] = l.getBounds()
            RP = R
            mult *= 1.05
            iter += 1
    else:
        while iter < 30:
            R = findOverlapRun(B)
            if not R:
                break
            nR = len(R)
            if nR == nL:
                break
            l1 = L[-1]
            orig1 = l1._origdata
            bounds1 = orig1['bounds']
            for i, r in enumerate(R):
                l = L[r]
                orig = l._origdata
                bounds = orig['bounds']
                diff1 = 0
                diff2 = 0
                if not i == nR - 1:
                    if not bounds == bounds1:
                        if bounds[3] > bounds1[1] and bounds1[1] < bounds[1]:
                            diff1 = bounds[3] - bounds1[1]
                        if bounds1[3] > bounds[1] and bounds[1] < bounds1[1]:
                            diff2 = bounds1[3] - bounds[1]
                        if diff1 > diff2:
                            l.y += 0.5 * (bounds1[3] - bounds1[1])
                        elif diff2 >= diff1:
                            l.y -= 0.5 * (bounds1[3] - bounds1[1])
                    B[r] = l.getBounds()
            iter += 1