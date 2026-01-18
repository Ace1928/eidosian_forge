import inspect
import weakref
from .Qt import QtCore, QtWidgets
def comboState(w):
    ind = w.currentIndex()
    data = w.itemData(ind)
    if data is not None:
        try:
            if not data.isValid():
                data = None
            else:
                data = data.toInt()[0]
        except AttributeError:
            pass
    if data is None:
        return str(w.itemText(ind))
    else:
        return data