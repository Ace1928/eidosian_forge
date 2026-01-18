from ..Qt import QtCore, QtWidgets
def insertChildren(self, index, childs):
    self._real_item.addChildren(index, childs)
    for child in childs:
        TreeWidget.informTreeWidgetChange(child)