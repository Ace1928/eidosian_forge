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
def sampleH1():
    """Sample of multi-series bar chart."""
    drawing = Drawing(400, 200)
    data = [(13, 5, 20, 22, 37, 45, 19, 4), (14, 6, 21, 23, 38, 46, 20, 5)]
    bc = HorizontalBarChart()
    bc.x = 50
    bc.y = 50
    bc.height = 125
    bc.width = 300
    bc.data = data
    bc.strokeColor = colors.black
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = 60
    bc.valueAxis.valueStep = 15
    bc.categoryAxis.labels.boxAnchor = 'e'
    catNames = 'Jan Feb Mar Apr May Jun Jul Aug'.split(' ')
    catNames = [n + '-99' for n in catNames]
    bc.categoryAxis.categoryNames = catNames
    drawing.add(bc, 'barchart')
    return drawing