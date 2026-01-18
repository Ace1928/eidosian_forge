from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def find_interval(lo, hi, I=5):
    """determine tick parameters for range [lo, hi] using I intervals"""
    if lo >= hi:
        if lo == hi:
            if lo == 0:
                lo = -0.1
                hi = 0.1
            else:
                lo = 0.9 * lo
                hi = 1.1 * hi
        else:
            raise ValueError('lo>hi')
    x = (hi - lo) / float(I)
    b = (x > 0 and (x < 1 or x > 10)) and 10 ** floor(log10(x)) or 1
    b = b
    while 1:
        a = x / b
        if a <= _intervals[-1]:
            break
        b = b * 10
    j = 0
    while a > _intervals[j]:
        j = j + 1
    while 1:
        ss = _intervals[j] * b
        n = lo / ss
        l = int(n) - (n < 0)
        n = ss * l
        x = ss * (l + I)
        a = I * ss
        if n > 0:
            if a >= hi:
                n = 0.0
                x = a
        elif hi < 0:
            a = -a
            if lo > a:
                n = a
                x = 0
        if hi <= x and n <= lo:
            break
        j = j + 1
        if j > _j_max:
            j = 0
            b = b * 10
    return (n, x, ss, lo - n + x - hi)