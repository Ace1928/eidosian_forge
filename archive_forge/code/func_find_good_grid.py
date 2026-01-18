from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def find_good_grid(lower, upper, n=(4, 5, 6, 7, 8, 9), grid=None):
    if grid:
        t = divmod(lower, grid)[0] * grid
        hi, z = divmod(upper, grid)
        if z > 1e-08:
            hi = hi + 1
        hi = hi * grid
    else:
        try:
            n[0]
        except TypeError:
            n = range(max(1, n - 2), max(n + 3, 2))
        w = 1e+308
        for i in n:
            z = find_interval(lower, upper, i)
            if z[3] < w:
                t, hi, grid = z[:3]
                w = z[3]
    return (t, hi, grid)