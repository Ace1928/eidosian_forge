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
def calcBarPositions(self):
    BarChart.calcBarPositions(self)
    seriesCount = self._seriesCount
    zDepth = self.zDepth
    if zDepth is None:
        zDepth = self.barWidth
    zSpace = self.zSpace
    if zSpace is None:
        zSpace = self.barSpacing
    if self.categoryAxis.style == 'parallel_3d':
        _3d_depth = seriesCount * zDepth + (seriesCount + 1) * zSpace
    else:
        _3d_depth = zDepth + 2 * zSpace
    _3d_depth *= self._normFactor
    self._3d_dx = self.theta_x * _3d_depth
    self._3d_dy = self.theta_y * _3d_depth