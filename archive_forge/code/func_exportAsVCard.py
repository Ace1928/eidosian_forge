import pickle
from PySide2 import QtCore, QtGui, QtWidgets
def exportAsVCard(self):
    name = str(self.nameLine.text())
    address = self.addressText.toPlainText()
    nameList = name.split()
    if len(nameList) > 1:
        firstName = nameList[0]
        lastName = nameList[-1]
    else:
        firstName = name
        lastName = ''
    fileName = QtWidgets.QFileDialog.getSaveFileName(self, 'Export Contact', '', 'vCard Files (*.vcf);;All Files (*)')[0]
    if not fileName:
        return
    out_file = QtCore.QFile(fileName)
    if not out_file.open(QtCore.QIODevice.WriteOnly):
        QtWidgets.QMessageBox.information(self, 'Unable to open file', out_file.errorString())
        return
    out_s = QtCore.QTextStream(out_file)
    out_s << 'BEGIN:VCARD' << '\n'
    out_s << 'VERSION:2.1' << '\n'
    out_s << 'N:' << lastName << ';' << firstName << '\n'
    out_s << 'FN:' << ' '.join(nameList) << '\n'
    address.replace(';', '\\;')
    address.replace('\n', ';')
    address.replace(',', ' ')
    out_s << 'ADR;HOME:;' << address << '\n'
    out_s << 'END:VCARD' << '\n'
    QtWidgets.QMessageBox.information(self, 'Export Successful', '"%s" has been exported as a vCard.' % name)