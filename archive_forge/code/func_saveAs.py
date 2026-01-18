from PySide2 import QtCore, QtGui, QtWidgets
import application_rc
def saveAs(self):
    fileName, filtr = QtWidgets.QFileDialog.getSaveFileName(self)
    if fileName:
        return self.saveFile(fileName)
    return False