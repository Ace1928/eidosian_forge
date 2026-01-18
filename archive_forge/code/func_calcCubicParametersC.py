from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
@cython.cfunc
@cython.inline
@cython.locals(pt1=cython.complex, pt2=cython.complex, pt3=cython.complex, pt4=cython.complex, a=cython.complex, b=cython.complex, c=cython.complex)
def calcCubicParametersC(pt1, pt2, pt3, pt4):
    c = (pt2 - pt1) * 3.0
    b = (pt3 - pt2) * 3.0 - c
    a = pt4 - pt1 - c - b
    return (a, b, c, pt1)