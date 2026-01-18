import sys
from PySide2 import QtCore, QtGui, QtWidgets
def informationMessage(self):
    reply = QtWidgets.QMessageBox.information(self, 'QMessageBox.information()', Dialog.MESSAGE)
    if reply == QtWidgets.QMessageBox.Ok:
        self.informationLabel.setText('OK')
    else:
        self.informationLabel.setText('Escape')