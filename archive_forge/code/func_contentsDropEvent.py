import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def contentsDropEvent(self, e):
    if e.mimeData().hasFormat('application/x-mycompany-VCard'):
        s = e.mimeData().data('application/x-mycompany-VCard')
        self.label2.setText(str(s))
        e.acceptProposedAction()