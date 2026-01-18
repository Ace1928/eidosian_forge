import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
def boundingBoxStr(self, x0, y0, x1, y1):
    """coordinates of bbox in default PS coordinates"""
    return '%%BoundingBox: ' + '%s %s %s %s' % (x0, y0, x1, y1)