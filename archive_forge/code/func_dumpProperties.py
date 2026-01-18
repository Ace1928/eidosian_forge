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
def dumpProperties(self, prefix=''):
    """Convenience. Lists them on standard output.  You
        may provide a prefix - mostly helps to generate code
        samples for documentation."""
    propList = list(self.getProperties().items())
    propList.sort()
    if prefix:
        prefix = prefix + '.'
    for name, value in propList:
        print('%s%s = %s' % (prefix, name, value))