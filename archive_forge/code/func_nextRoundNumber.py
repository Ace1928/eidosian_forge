from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def nextRoundNumber(x):
    """Return the first 'nice round number' greater than or equal to x

    Used in selecting apropriate tick mark intervals; we say we want
    an interval which places ticks at least 10 points apart, work out
    what that is in chart space, and ask for the nextRoundNumber().
    Tries the series 1,2,5,10,20,50,100.., going up or down as needed.
    """
    if x in (0, 1):
        return x
    if x < 0:
        return -1.0 * nextRoundNumber(-x)
    else:
        lg = int(log10(x))
        if lg == 0:
            if x < 1:
                base = 0.1
            else:
                base = 1.0
        elif lg < 0:
            base = 10.0 ** (lg - 1)
        else:
            base = 10.0 ** lg
        if base >= x:
            return base * 1.0
        elif base * 2 >= x:
            return base * 2.0
        elif base * 5 >= x:
            return base * 5.0
        else:
            return base * 10.0