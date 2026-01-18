import os, sys
from math import pi, cos, sin, sqrt, radians, floor
from reportlab.platypus import Flowable
from reportlab.rl_config import shapeChecking, verbose, defaultGraphicsFontName as _baseGFontName, _unset_, decimalSymbol
from reportlab.lib import logger
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.utils import isSeq, asBytes
from reportlab.lib.attrmap import *
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.fonts import tt2ps
from reportlab.pdfgen.canvas import FILL_EVEN_ODD, FILL_NON_ZERO
from . transform import *
def getRectsBounds(rectList):
    L = [x for x in rectList if x is not None]
    if not L:
        return None
    xMin, yMin, xMax, yMax = L[0]
    for x1, y1, x2, y2 in L[1:]:
        if x1 < xMin:
            xMin = x1
        if x2 > xMax:
            xMax = x2
        if y1 < yMin:
            yMin = y1
        if y2 > yMax:
            yMax = y2
    return (xMin, yMin, xMax, yMax)