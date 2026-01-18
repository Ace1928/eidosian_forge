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
def definePath(pathSegs=[], isClipPath=0, dx=0, dy=0, **kw):
    O = []
    P = []
    for seg in pathSegs:
        if not isSeq(seg):
            opName = seg
            args = []
        else:
            opName = seg[0]
            args = seg[1:]
        if opName not in _PATH_OP_NAMES:
            raise ValueError('bad operator name %s' % opName)
        op = _PATH_OP_NAMES.index(opName)
        if len(args) != _PATH_OP_ARG_COUNT[op]:
            raise ValueError('%s bad arguments %s' % (opName, str(args)))
        O.append(op)
        P.extend(list(args))
    for d, o in ((dx, 0), (dy, 1)):
        for i in range(o, len(P), 2):
            P[i] = P[i] + d
    bbox = kw.pop('bbox', None)
    if bbox:
        for j in (0, 1):
            d = (bbox[j], bbox[j + 2])
            if d[0] is None and d[1] is None:
                continue
            a = P[j::2]
            a, b = (min(a), max(a))
            if d[0] is not None and d[1] is not None:
                c, d = (min(d), max(d))
                fac = b - a
                if abs(fac) >= 1e-06:
                    fac = (d - c) / fac
                    for i in range(j, len(P), 2):
                        P[i] = c + fac * (P[i] - a)
                else:
                    for i in range(j, len(P), 2):
                        P[i] = (c + d) * 0.5
            else:
                c = d[0] - a if d[0] is not None else d[1] - b
                for i in range(j, len(P), 2):
                    P[i] += c
    return Path(P, O, isClipPath, **kw)