import decimal
import re
import warnings
from math import isinf, isnan
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
def focusInEvent(self, ev):
    super(SpinBox, self).focusInEvent(ev)
    self.selectNumber()