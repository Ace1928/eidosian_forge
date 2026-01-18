import sys
from PySide2 import QtCore, QtGui, QtWidgets
def criticalMessage(self):
    reply = QtWidgets.QMessageBox.critical(self, 'QMessageBox.critical()', Dialog.MESSAGE, QtWidgets.QMessageBox.Abort | QtWidgets.QMessageBox.Retry | QtWidgets.QMessageBox.Ignore)
    if reply == QtWidgets.QMessageBox.Abort:
        self.criticalLabel.setText('Abort')
    elif reply == QtWidgets.QMessageBox.Retry:
        self.criticalLabel.setText('Retry')
    else:
        self.criticalLabel.setText('Ignore')