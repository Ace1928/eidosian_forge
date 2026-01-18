from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfNumbersOrNone,\
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Line, Rect, Polygon, PolyLine, \
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab.graphics.charts.piecharts import WedgeLabel
from reportlab.graphics.widgets.markers import makeMarker, uSymbol2Symbol, isSymbol
def labelClass(self, kind):
    klass = getattr(self, f'{kind}LabelClass', None)
    if not klass:
        klass = globals()[f'{kind.capitalize()}Label']
    return klass