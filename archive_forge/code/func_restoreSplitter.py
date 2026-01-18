import inspect
import weakref
from .Qt import QtCore, QtWidgets
def restoreSplitter(w, s):
    if type(s) is list:
        w.setSizes(s)
    elif type(s) is str:
        w.restoreState(QtCore.QByteArray.fromPercentEncoding(s.encode()))
    else:
        print("Can't configure QSplitter using object of type", type(s))
    if w.count() > 0:
        for i in w.sizes():
            if i > 0:
                return
        w.setSizes([50] * w.count())