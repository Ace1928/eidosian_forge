import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
def focusOutEvent(self, event):
    self.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
    self.lostFocus.emit(self)
    super(DiagramTextItem, self).focusOutEvent(event)