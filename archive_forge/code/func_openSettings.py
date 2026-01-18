import sys
from PySide2 import QtCore, QtGui, QtWidgets
def openSettings(self):
    if self.locationDialog is None:
        self.locationDialog = LocationDialog(self)
    if self.locationDialog.exec_():
        settings = QtCore.QSettings(self.locationDialog.format(), self.locationDialog.scope(), self.locationDialog.organization(), self.locationDialog.application())
        self.setSettingsObject(settings)
        self.fallbacksAct.setEnabled(True)