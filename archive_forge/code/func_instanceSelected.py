from PySide2 import QtCore, QtGui, QtWidgets, QtXmlPatterns
import schema_rc
from ui_schema import Ui_SchemaMainWindow
def instanceSelected(self, index):
    if index is -1:
        return
    index += 2 * self.schemaSelection.currentIndex()
    instanceFile = QtCore.QFile(':/instance_%d.xml' % index)
    instanceFile.open(QtCore.QIODevice.ReadOnly)
    instanceData = instanceFile.readAll()
    self.instanceEdit.setPlainText(encode_utf8(instanceData))
    self.validate()