from ..Qt import QtCore, QtWidgets
def setChecked(self, column, checked):
    self.setCheckState(column, QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)