from ..Qt import QtCore, QtWidgets
def setExpanded(self, exp):
    self._expanded = exp
    QtWidgets.QTreeWidgetItem.setExpanded(self, exp)