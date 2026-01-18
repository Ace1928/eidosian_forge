from time import perf_counter
from ..Qt import QtCore, QtGui, QtWidgets
def setLabelText(self, val):
    if self.disabled:
        return
    QtWidgets.QProgressDialog.setLabelText(self, val)