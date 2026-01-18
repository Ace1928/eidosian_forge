from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def maverage(data, n=6):
    data = (n - 1) * [data[0]] + data
    data = [float(sum(data[i - n:i])) / n for i in range(n, len(data) + 1)]
    return data