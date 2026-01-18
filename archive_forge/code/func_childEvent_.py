import weakref
from ..Qt import QtCore, QtWidgets
from .Dock import Dock
def childEvent_(self, ev):
    ch = ev.child()
    if ev.removed() and hasattr(ch, 'sigStretchChanged'):
        try:
            ch.sigStretchChanged.disconnect(self.childStretchChanged)
        except:
            pass
        self.updateStretch()