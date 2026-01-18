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
def resized(self, kind='fit', lpad=0, rpad=0, bpad=0, tpad=0):
    """return a base class drawing which ensures all the contents fits"""
    C = self.getContents()
    oW = self.width
    oH = self.height
    drawing = Drawing(oW, oH, *C)
    xL, yL, xH, yH = drawing.getBounds()
    if kind == 'fit' or (kind == 'expand' and (xL < lpad or xH > oW - rpad or yL < bpad or (yH > oH - tpad))):
        drawing.width = xH - xL + lpad + rpad
        drawing.height = yH - yL + tpad + bpad
        drawing.transform = (1, 0, 0, 1, lpad - xL, bpad - yL)
    elif kind == 'fitx' or (kind == 'expandx' and (xL < lpad or xH > oW - rpad)):
        drawing.width = xH - xL + lpad + rpad
        drawing.transform = (1, 0, 0, 1, lpad - xL, 0)
    elif kind == 'fity' or (kind == 'expandy' and (yL < bpad or yH > oH - tpad)):
        drawing.height = yH - yL + tpad + bpad
        drawing.transform = (1, 0, 0, 1, 0, bpad - yL)
    return drawing