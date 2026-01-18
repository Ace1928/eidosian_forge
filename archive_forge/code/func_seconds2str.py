from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def seconds2str(seconds):
    """Convert a date string into the number of seconds since the epoch."""
    return strftime('%Y-%m-%d', gmtime(seconds))