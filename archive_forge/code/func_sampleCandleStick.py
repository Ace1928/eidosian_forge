from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColorOrNone, \
from reportlab.lib.attrmap import *
from reportlab.lib.utils import flatten
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.shapes import Line, Rect, Group, Drawing, Polygon, PolyLine
from reportlab.graphics.widgets.signsandsymbols import NoEntry
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol, makeMarker
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from .utils import FillPairedData
def sampleCandleStick():
    from reportlab.graphics.widgetbase import CandleSticks
    d = Drawing(400, 200)
    chart = HorizontalLineChart()
    d.add(chart)
    chart.y = 20
    boxMid = (100, 110, 120, 130)
    hi = [m + 10 for m in boxMid]
    lo = [m - 10 for m in boxMid]
    boxHi = [m + 6 for m in boxMid]
    boxLo = [m - 4 for m in boxMid]
    boxFillColor = colors.pink
    boxWidth = 20
    crossWidth = 10
    candleStrokeWidth = 0.5
    candleStrokeColor = colors.black
    chart.valueAxis.avoidBoundSpace = 5
    chart.valueAxis.valueMin = min(min(boxMid), min(hi), min(lo), min(boxLo), min(boxHi))
    chart.valueAxis.valueMax = max(max(boxMid), max(hi), max(lo), max(boxLo), max(boxHi))
    lines = chart.lines
    lines[0].strokeColor = None
    I = range(len(boxMid))
    chart.data = [boxMid]
    lines[0].symbol = candles = CandleSticks(chart=chart, boxFillColor=boxFillColor, boxWidth=boxWidth, crossWidth=crossWidth, strokeWidth=candleStrokeWidth, strokeColor=candleStrokeColor)
    for i in I:
        candles[i].setProperties(dict(position=i, boxMid=boxMid[i], crossLo=lo[i], crossHi=hi[i], boxLo=boxLo[i], boxHi=boxHi[i]))
    return d