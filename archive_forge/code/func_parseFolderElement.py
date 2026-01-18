from PySide2 import QtCore, QtGui, QtWidgets, QtXml
def parseFolderElement(self, element, parentItem=None):
    item = self.createItem(element, parentItem)
    title = element.firstChildElement('title').text()
    if not title:
        title = 'Folder'
    item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
    item.setIcon(0, self.folderIcon)
    item.setText(0, title)
    folded = element.attribute('folded') != 'no'
    self.setItemExpanded(item, not folded)
    child = element.firstChildElement()
    while not child.isNull():
        if child.tagName() == 'folder':
            self.parseFolderElement(child, item)
        elif child.tagName() == 'bookmark':
            childItem = self.createItem(child, item)
            title = child.firstChildElement('title').text()
            if not title:
                title = 'Folder'
            childItem.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
            childItem.setIcon(0, self.bookmarkIcon)
            childItem.setText(0, title)
            childItem.setText(1, child.attribute('href'))
        elif child.tagName() == 'separator':
            childItem = self.createItem(child, item)
            childItem.setFlags(item.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
            childItem.setText(0, 30 * '·')
        child = child.nextSiblingElement()