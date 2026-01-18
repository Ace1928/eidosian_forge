from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import flatten, isStr
from reportlab.graphics.shapes import Drawing, Group, Rect, PolyLine, Polygon, _SetKeyWordArgs
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.charts.axes import XValueAxis, YValueAxis, AdjYValueAxis, NormalDateXValueAxis
from reportlab.graphics.charts.utils import *
from reportlab.graphics.widgets.markers import uSymbol2Symbol, makeMarker
from reportlab.graphics.widgets.grids import Grid, DoubleGrid, ShadedPolygon
from reportlab.pdfbase.pdfmetrics import stringWidth, getFont
from reportlab.graphics.charts.areas import PlotArea
from .utils import FillPairedData
from reportlab.graphics.charts.linecharts import AbstractLineChart
def sample2():
    """A line plot with non-equidistant points in x-axis."""
    drawing = Drawing(400, 200)
    data = [(('25/11/1991', 1), ('30/11/1991', 1.000933333), ('31/12/1991', 1.0062), ('31/01/1992', 1.0112), ('29/02/1992', 1.0158), ('31/03/1992', 1.020733333), ('30/04/1992', 1.026133333), ('31/05/1992', 1.030266667), ('30/06/1992', 1.034466667), ('31/07/1992', 1.038733333), ('31/08/1992', 1.0422), ('30/09/1992', 1.045533333), ('31/10/1992', 1.049866667), ('30/11/1992', 1.054733333), ('31/12/1992', 1.061))]
    data[0] = preprocessData(data[0])
    lp = LinePlot()
    lp.x = 50
    lp.y = 50
    lp.height = 125
    lp.width = 300
    lp.data = data
    lp.joinedLines = 1
    lp.lines.symbol = makeMarker('FilledDiamond')
    lp.strokeColor = colors.black
    start = mktime(mkTimeTuple('25/11/1991'))
    t0 = mktime(mkTimeTuple('30/11/1991'))
    t1 = mktime(mkTimeTuple('31/12/1991'))
    t2 = mktime(mkTimeTuple('31/03/1992'))
    t3 = mktime(mkTimeTuple('30/06/1992'))
    t4 = mktime(mkTimeTuple('30/09/1992'))
    end = mktime(mkTimeTuple('31/12/1992'))
    lp.xValueAxis.valueMin = start
    lp.xValueAxis.valueMax = end
    lp.xValueAxis.valueSteps = [start, t0, t1, t2, t3, t4, end]
    lp.xValueAxis.labelTextFormat = seconds2str
    lp.xValueAxis.labels[1].dy = -20
    lp.xValueAxis.labels[2].dy = -35
    lp.yValueAxis.labelTextFormat = '%4.2f'
    lp.yValueAxis.valueMin = 100
    lp.yValueAxis.valueMax = 110
    lp.yValueAxis.valueStep = 2
    drawing.add(lp)
    return drawing