import decimal
import re
import warnings
from math import isinf, isnan
from .. import functions as fn
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
def editingFinishedEvent(self):
    """Edit has finished; set value."""
    if self.lineEdit().text() == self.lastText:
        return
    try:
        val = self.interpret()
    except Exception:
        return
    if val is False:
        return
    if val == self.val:
        return
    self.setValue(val, delaySignal=False)