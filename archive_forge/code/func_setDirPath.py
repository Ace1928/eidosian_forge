from PySide2 import QtCore, QtGui, QtWidgets
def setDirPath(self, path):
    dir = QtCore.QDir(path)
    self.beginResetModel()
    self.fileList = list(dir.entryList())
    self.fileCount = 0
    self.endResetModel()