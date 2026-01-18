from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import open_and_read, isBytes, rl_exec
from .shapes import _baseGFontName, _PATH_OP_ARG_COUNT, _PATH_OP_NAMES, definePath
from sys import exc_info
def conic_to(a, b, ctx):
    """using the cubic equivalent"""
    x0, y0 = P[-1][-2:] if P else (a.x, a.y)
    x1 = xpt(a.x)
    y1 = ypt(a.y)
    x2 = xpt(b.x)
    y2 = ypt(b.y)
    P_append(('curveTo', x0 + (x1 - x0) * 2 / 3, y0 + (y1 - y0) * 2 / 3, x1 + (x2 - x1) / 3, y1 + (y2 - y1) / 3, x2, y2))