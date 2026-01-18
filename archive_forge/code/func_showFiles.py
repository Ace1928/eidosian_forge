from PySide2 import QtCore, QtGui, QtWidgets
def showFiles(self, files):
    for fn in files:
        file = QtCore.QFile(self.currentDir.absoluteFilePath(fn))
        size = QtCore.QFileInfo(file).size()
        fileNameItem = QtWidgets.QTableWidgetItem(fn)
        fileNameItem.setFlags(fileNameItem.flags() ^ QtCore.Qt.ItemIsEditable)
        sizeItem = QtWidgets.QTableWidgetItem('%d KB' % int((size + 1023) / 1024))
        sizeItem.setTextAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        sizeItem.setFlags(sizeItem.flags() ^ QtCore.Qt.ItemIsEditable)
        row = self.filesTable.rowCount()
        self.filesTable.insertRow(row)
        self.filesTable.setItem(row, 0, fileNameItem)
        self.filesTable.setItem(row, 1, sizeItem)
    self.filesFoundLabel.setText('%d file(s) found (Double click on a file to open it)' % len(files))