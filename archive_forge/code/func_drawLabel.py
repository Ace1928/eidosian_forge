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
def drawLabel(self, G, rowNo, colNo, x, y):
    """Draw a label for a given item in the list.
        G must have an add method"""
    G.add(self._innerDrawLabel(rowNo, colNo, x, y))