from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfStringsOrNone, OneOf,\
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Wedge
from reportlab.graphics.widgetbase import TypedPropertyCollection
from reportlab.graphics.charts.piecharts import AbstractPieChart, WedgeProperties, _addWedgeLabel, fixLabelOverlaps
from functools import reduce
def sample3():
    """Make a more complex demo"""
    d = Drawing(400, 400)
    dn = Doughnut()
    dn.x = 50
    dn.y = 50
    dn.width = 300
    dn.height = 300
    dn.data = [[10, 20, 30, 40, 50, 60], [10, 20, 30, 40]]
    dn.labels = ['a', 'b', 'c', 'd', 'e', 'f']
    d.add(dn)
    return d