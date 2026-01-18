from PySide2 import QtCore, QtGui, QtWidgets, QtSvg
def insertTextObject(self):
    fileName = self.fileNameLineEdit.text()
    file = QtCore.QFile(fileName)
    if not file.open(QtCore.QIODevice.ReadOnly):
        QtWidgets.QMessageBox.warning(self, self.tr('Error Opening File'), self.tr("Could not open '%1'").arg(fileName))
    svgData = file.readAll()
    svgCharFormat = QtGui.QTextCharFormat()
    svgCharFormat.setObjectType(Window.SvgTextFormat)
    svgCharFormat.setProperty(Window.SvgData, svgData)
    cursor = self.textEdit.textCursor()
    cursor.insertText(u'�', svgCharFormat)
    self.textEdit.setTextCursor(cursor)