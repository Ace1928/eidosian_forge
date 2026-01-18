from PySide2 import QtCore, QtGui, QtWidgets, QtXmlPatterns
import schema_rc
from ui_schema import Ui_SchemaMainWindow
def schemaSelected(self, index):
    self.instanceSelection.clear()
    if index == 0:
        self.instanceSelection.addItem('Valid Contact Instance')
        self.instanceSelection.addItem('Invalid Contact Instance')
    elif index == 1:
        self.instanceSelection.addItem('Valid Recipe Instance')
        self.instanceSelection.addItem('Invalid Recipe Instance')
    elif index == 2:
        self.instanceSelection.addItem('Valid Order Instance')
        self.instanceSelection.addItem('Invalid Order Instance')
    self.textChanged()
    schemaFile = QtCore.QFile(':/schema_%d.xsd' % index)
    schemaFile.open(QtCore.QIODevice.ReadOnly)
    schemaData = schemaFile.readAll()
    self.schemaView.setPlainText(encode_utf8(schemaData))
    self.validate()