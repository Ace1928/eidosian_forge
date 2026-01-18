import warnings
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop
def dragMoveEvent(self, *args):
    self.dockdrop.dragMoveEvent(*args)