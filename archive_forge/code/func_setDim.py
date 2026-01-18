import warnings
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop
def setDim(self, d):
    if self.dim != d:
        self.dim = d
        self.updateStyle()