import copy, functools
from ast import literal_eval
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColorOrNone, isString,\
from reportlab.lib.utils import isStr, yieldNoneSplits
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.shapes import Line, Rect, Group, Drawing, PolyLine
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis, YCategoryAxis, XValueAxis
from reportlab.graphics.charts.textlabels import BarChartLabel, NoneOrInstanceOfNA_Label
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab import cmp
def makeSwatchSample(self, rowNo, x, y, width, height):
    baseStyle = self.bars
    styleIdx = rowNo % len(baseStyle)
    style = baseStyle[styleIdx]
    strokeColor = getattr(style, 'strokeColor', getattr(baseStyle, 'strokeColor', None))
    fillColor = getattr(style, 'fillColor', getattr(baseStyle, 'fillColor', None))
    strokeDashArray = getattr(style, 'strokeDashArray', getattr(baseStyle, 'strokeDashArray', None))
    strokeWidth = getattr(style, 'strokeWidth', getattr(style, 'strokeWidth', None))
    swatchMarker = getattr(style, 'swatchMarker', getattr(baseStyle, 'swatchMarker', None))
    if swatchMarker:
        return uSymbol2Symbol(swatchMarker, x + width / 2.0, y + height / 2.0, fillColor)
    elif getattr(style, 'isLine', False):
        yh2 = y + height / 2.0
        if hasattr(style, 'symbol'):
            S = style.symbol
        elif hasattr(baseStyle, 'symbol'):
            S = baseStyle.symbol
        else:
            S = None
        L = Line(x, yh2, x + width, yh2, strokeColor=style.strokeColor or style.fillColor, strokeWidth=style.strokeWidth, strokeDashArray=style.strokeDashArray)
        if S:
            S = uSymbol2Symbol(S, x + width / 2.0, yh2, style.strokeColor or style.fillColor)
        if S and L:
            g = Group()
            g.add(L)
            g.add(S)
            return g
        return S or L
    else:
        return Rect(x, y, width, height, strokeWidth=strokeWidth, strokeColor=strokeColor, strokeDashArray=strokeDashArray, fillColor=fillColor)