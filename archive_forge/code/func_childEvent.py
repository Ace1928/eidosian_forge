import weakref
from ..Qt import QtCore, QtWidgets
from .Dock import Dock
def childEvent(self, ev):
    super().childEvent(ev)
    self.container.childEvent_(ev)