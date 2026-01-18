from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
@staticmethod
def rectDrawTimeCallback(node, canvas, renderer, **kwds):
    A = getattr(canvas, 'ctm', None)
    if not A:
        return
    x1 = node.x
    y1 = node.y
    x2 = x1 + node.width
    y2 = y1 + node.height
    D = kwds.copy()
    D['rect'] = DrawTimeCollector.transformAndFlatten(A, ((x1, y1), (x2, y2)))
    return D