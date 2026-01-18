import sys
from PySide2 import QtCore, QtGui, QtWidgets
def setSaveFileName(self):
    options = QtWidgets.QFileDialog.Options()
    if not self.native.isChecked():
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
    fileName, filtr = QtWidgets.QFileDialog.getSaveFileName(self, 'QFileDialog.getSaveFileName()', self.saveFileNameLabel.text(), 'All Files (*);;Text Files (*.txt)', '', options)
    if fileName:
        self.saveFileNameLabel.setText(fileName)