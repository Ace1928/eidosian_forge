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
def sampleSymbol1():
    """Simple bar chart using symbol attribute."""
    drawing = Drawing(400, 200)
    data = dataSample5
    bc = VerticalBarChart()
    bc.x = 50
    bc.y = 50
    bc.height = 125
    bc.width = 300
    bc.data = data
    bc.strokeColor = colors.black
    bc.barWidth = 10
    bc.groupSpacing = 15
    bc.barSpacing = 3
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = 60
    bc.valueAxis.valueStep = 15
    bc.categoryAxis.labels.boxAnchor = 'e'
    bc.categoryAxis.categoryNames = ['Ying', 'Yang']
    from reportlab.graphics.widgets.grids import ShadedRect
    sym1 = ShadedRect()
    sym1.fillColorStart = colors.black
    sym1.fillColorEnd = colors.blue
    sym1.orientation = 'horizontal'
    sym1.strokeWidth = 0
    sym2 = ShadedRect()
    sym2.fillColorStart = colors.black
    sym2.fillColorEnd = colors.pink
    sym2.orientation = 'horizontal'
    sym2.strokeWidth = 0
    sym3 = ShadedRect()
    sym3.fillColorStart = colors.blue
    sym3.fillColorEnd = colors.white
    sym3.orientation = 'vertical'
    sym3.cylinderMode = 1
    sym3.strokeWidth = 0
    bc.bars.symbol = sym1
    bc.bars[2].symbol = sym2
    bc.bars[3].symbol = sym3
    drawing.add(bc)
    return drawing