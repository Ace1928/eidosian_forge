import sys
from PySide2 import QtCore, QtGui, QtWidgets
def setOpenFileNames(self):
    options = QtWidgets.QFileDialog.Options()
    if not self.native.isChecked():
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
    files, filtr = QtWidgets.QFileDialog.getOpenFileNames(self, 'QFileDialog.getOpenFileNames()', self.openFilesPath, 'All Files (*);;Text Files (*.txt)', '', options)
    if files:
        self.openFilesPath = files[0]
        self.openFileNamesLabel.setText('[%s]' % ', '.join(files))