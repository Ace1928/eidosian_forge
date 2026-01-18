import warnings
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.VerticalLabel import VerticalLabel
from .DockDrop import DockDrop
def startDrag(self):
    self.drag = QtGui.QDrag(self)
    mime = QtCore.QMimeData()
    self.drag.setMimeData(mime)
    self.widgetArea.setStyleSheet(self.dragStyle)
    self.update()
    action = self.drag.exec() if hasattr(self.drag, 'exec') else self.drag.exec_()
    self.updateStyle()