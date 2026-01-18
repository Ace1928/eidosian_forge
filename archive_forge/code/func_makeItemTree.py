import sys, traceback
from ..Qt import QtWidgets, QtGui
def makeItemTree(stack, title):
    topItem = QtWidgets.QTreeWidgetItem([title])
    topItem.frame = None
    font = topItem.font(0)
    font.setWeight(font.Weight.Bold)
    topItem.setFont(0, font)
    items = [topItem]
    for entry in stack:
        if isinstance(entry, QtWidgets.QTreeWidgetItem):
            item = entry
        else:
            text, frame = entry
            item = QtWidgets.QTreeWidgetItem([text.rstrip()])
            item.frame = frame
        topItem.addChild(item)
        items.append(item)
    return items