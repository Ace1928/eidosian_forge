from ..Qt import QtCore, QtWidgets
def collapseTree(self, item):
    item.setExpanded(False)
    for i in range(item.childCount()):
        self.collapseTree(item.child(i))