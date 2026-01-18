from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isColorOrNone, isBoolean, isListOfNumbers, OneOf, isListOfColors, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, LineShape, definePath, EmptyClipPath
from reportlab.graphics.widgetbase import Widget
def centroid(P):
    """compute average point of a set of points"""
    cx = 0
    cy = 0
    for x, y in P:
        cx += x
        cy += y
    n = float(len(P))
    return (cx / n, cy / n)