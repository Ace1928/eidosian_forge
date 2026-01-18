from ..Qt import QtCore, QtWidgets
def removeWidget(self, column):
    del self._widgets[column]
    tree = self.treeWidget()
    if tree is None:
        return
    tree.removeItemWidget(self, column)