from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
@property
def pmcanv(self):
    if not self._pmcanv:
        import renderPM
        self._pmcanv = renderPM.PMCanvas(1, 1)
    return self._pmcanv