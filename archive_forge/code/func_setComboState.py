import inspect
import weakref
from .Qt import QtCore, QtWidgets
def setComboState(w, v):
    if type(v) is int:
        ind = w.findData(v)
        if ind > -1:
            w.setCurrentIndex(ind)
            return
    w.setCurrentIndex(w.findText(str(v)))