from ..Qt import QtCore, QtWidgets
from . import VerticalLabel
def removeRow(self, name):
    row = self.rowNames.index(name)
    self.oldRows[name] = self.saveState()['rows'][row]
    self.rowNames.pop(row)
    for w in self.rowWidgets[row]:
        w.setParent(None)
        if isinstance(w, QtWidgets.QCheckBox):
            w.stateChanged.disconnect(self.checkChanged)
    self.rowWidgets.pop(row)
    for i in range(row, len(self.rowNames)):
        widgets = self.rowWidgets[i]
        for j in range(len(widgets)):
            widgets[j].setParent(None)
            self.layout.addWidget(widgets[j], i + 1, j)