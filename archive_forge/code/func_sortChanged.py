from PySide2 import QtCore, QtGui, QtWidgets
def sortChanged(self):
    if self.sortCaseSensitivityCheckBox.isChecked():
        caseSensitivity = QtCore.Qt.CaseSensitive
    else:
        caseSensitivity = QtCore.Qt.CaseInsensitive
    self.proxyModel.setSortCaseSensitivity(caseSensitivity)