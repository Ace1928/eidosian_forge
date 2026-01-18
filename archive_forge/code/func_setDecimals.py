import decimal
import re
import warnings
from math import isinf, isnan
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
def setDecimals(self, decimals):
    """Set the number of decimals to be displayed when formatting numeric
        values.
        """
    self.setOpts(decimals=decimals)