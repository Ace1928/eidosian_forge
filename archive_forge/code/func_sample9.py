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
def sample9():
    """Case with overlapping labels"""
    'Labels overlap if they do not belong to adjacent pies due to nature of checkLabelOverlap'
    d = Drawing(400, 200)
    pc = Pie()
    pc.x = 125
    pc.y = 50
    pc.data = [41, 20, 40, 15, 20, 30, 50, 15, 25, 35, 25, 20, 30, 40, 20, 30]
    pc.labels = ['example1', 'example2', 'example3', 'example4', 'example5', 'example6', 'example7', 'example8', 'example9', 'example10', 'example11', 'example12', 'example13', 'example14', 'example15', 'example16']
    pc.sideLabels = 1
    pc.checkLabelOverlap = 1
    pc.width = 100
    pc.height = 100
    pc.slices.strokeWidth = 1
    pc.slices[0].fillColor = colors.steelblue
    pc.slices[1].fillColor = colors.thistle
    pc.slices[2].fillColor = colors.cornflower
    pc.slices[3].fillColor = colors.lightsteelblue
    pc.slices[4].fillColor = colors.aquamarine
    pc.slices[5].fillColor = colors.cadetblue
    d.add(pc)
    return d