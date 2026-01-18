from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
def calcQuadraticPoints(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    x1 = cx
    y1 = cy
    x2 = bx * 0.5 + cx
    y2 = by * 0.5 + cy
    x3 = ax + bx + cx
    y3 = ay + by + cy
    return ((x1, y1), (x2, y2), (x3, y3))