import warnings
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop
def containerChanged(self, c):
    if self._container is not None:
        self._container.apoptose(propagate=False)
    self._container = c
    if c is None:
        self.area = None
    else:
        self.area = c.area
        if c.type() != 'tab':
            self.moveLabel = True
            self.label.setDim(False)
        else:
            self.moveLabel = False
        self.setOrientation(force=True)