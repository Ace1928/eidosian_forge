from ..Qt import QtCore, QtWidgets
def isChecked(self, col):
    return self.checkState(col) == QtCore.Qt.CheckState.Checked