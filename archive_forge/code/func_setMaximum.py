from time import perf_counter
from ..Qt import QtCore, QtGui, QtWidgets
def setMaximum(self, val):
    if self.disabled:
        return
    QtWidgets.QProgressDialog.setMaximum(self, val)