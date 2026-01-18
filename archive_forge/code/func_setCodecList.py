from PySide2 import QtCore, QtGui, QtWidgets
def setCodecList(self, codecs):
    self.encodingComboBox.clear()
    for codec in codecs:
        self.encodingComboBox.addItem(codec_name(codec), codec.mibEnum())