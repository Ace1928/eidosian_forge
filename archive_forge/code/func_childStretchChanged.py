import weakref
from ..Qt import QtCore, QtWidgets
from .Dock import Dock
def childStretchChanged(self):
    self.updateStretch()