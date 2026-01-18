from PySide2 import QtCore, QtGui, QtWidgets
import application_rc
def maybeSave(self):
    if self.textEdit.document().isModified():
        ret = QtWidgets.QMessageBox.warning(self, 'Application', 'The document has been modified.\nDo you want to save your changes?', QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
        if ret == QtWidgets.QMessageBox.Save:
            return self.save()
        elif ret == QtWidgets.QMessageBox.Cancel:
            return False
    return True