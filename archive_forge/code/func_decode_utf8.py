from PySide2 import QtCore, QtGui, QtWidgets, QtXmlPatterns
import schema_rc
from ui_schema import Ui_SchemaMainWindow
def decode_utf8(qs):
    return QtCore.QByteArray(bytes(qs, encoding='utf8'))