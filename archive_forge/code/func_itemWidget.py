from ..Qt import QtCore, QtWidgets
def itemWidget(self, item, col):
    w = QtWidgets.QTreeWidget.itemWidget(self, item, col)
    if w is not None and hasattr(w, 'realChild'):
        w = w.realChild
    return w