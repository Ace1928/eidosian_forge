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
def calcPositions(self):
    LinePlot.calcPositions(self)
    nSeries = self._seriesCount
    zSpace = self.zSpace
    zDepth = self.zDepth
    if self.xValueAxis.style == 'parallel_3d':
        _3d_depth = nSeries * zDepth + (nSeries + 1) * zSpace
    else:
        _3d_depth = zDepth + 2 * zSpace
    self._3d_dx = self.theta_x * _3d_depth
    self._3d_dy = self.theta_y * _3d_depth