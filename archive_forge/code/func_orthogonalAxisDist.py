from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isColorOrNone, isBoolean, isListOfNumbers, OneOf, isListOfColors, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, LineShape, definePath, EmptyClipPath
from reportlab.graphics.widgetbase import Widget
def orthogonalAxisDist(xy, s=s, c=c, x0=x0, y0=y0):
    x, y = xy
    return c * (y - y0) + s * (x - x0)