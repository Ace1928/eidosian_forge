import numpy as np
from .. import functions as fn
from ..Qt import QtWidgets
from .DataTreeWidget import DataTreeWidget
def compareArrays(self, a, b):
    intnan = -9223372036854775808
    anans = np.isnan(a) | (a == intnan)
    bnans = np.isnan(b) | (b == intnan)
    eq = anans == bnans
    mask = ~anans
    eq[mask] = np.allclose(a[mask], b[mask])
    return eq