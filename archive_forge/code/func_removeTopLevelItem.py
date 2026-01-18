from ..Qt import QtCore, QtWidgets
def removeTopLevelItem(self, item):
    for i in range(self.topLevelItemCount()):
        if self.topLevelItem(i) is item:
            self.takeTopLevelItem(i)
            return
    raise Exception("Item '%s' not in top-level items." % str(item))