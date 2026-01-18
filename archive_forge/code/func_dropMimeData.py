from ..Qt import QtCore, QtWidgets
def dropMimeData(self, parent, index, data, action):
    item = self.currentItem()
    p = parent
    while True:
        if p is None:
            break
        if p is item:
            return False
        p = p.parent()
    if not self.itemMoving(item, parent, index):
        return False
    currentParent = item.parent()
    if currentParent is None:
        currentParent = self.invisibleRootItem()
    if parent is None:
        parent = self.invisibleRootItem()
    if currentParent is parent and index > parent.indexOfChild(item):
        index -= 1
    self.prepareMove(item)
    currentParent.removeChild(item)
    parent.insertChild(index, item)
    self.setCurrentItem(item)
    self.recoverMove(item)
    self.sigItemMoved.emit(item, parent, index)
    return True