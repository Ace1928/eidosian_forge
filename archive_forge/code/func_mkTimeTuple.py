from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def mkTimeTuple(timeString):
    """Convert a 'dd/mm/yyyy' formatted string to a tuple for use in the time module."""
    L = [0] * 9
    dd, mm, yyyy = list(map(int, timeString.split('/')))
    L[:3] = [yyyy, mm, dd]
    return tuple(L)