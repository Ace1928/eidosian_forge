import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def logTickStrings(self, values, scale, spacing):
    estrings = ['%0.1g' % x for x in 10 ** np.array(values).astype(float) * np.array(scale)]
    convdict = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
    dstrings = []
    for e in estrings:
        if e.count('e'):
            v, p = e.split('e')
            sign = '⁻' if p[0] == '-' else ''
            pot = ''.join([convdict[pp] for pp in p[1:].lstrip('0')])
            if v == '1':
                v = ''
            else:
                v = v + '·'
            dstrings.append(v + '10' + sign + pot)
        else:
            dstrings.append(e)
    return dstrings