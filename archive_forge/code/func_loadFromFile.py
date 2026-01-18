import pickle
from PySide2 import QtCore, QtGui, QtWidgets
def loadFromFile(self):
    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Address Book', '', 'Address Book (*.abk);;All Files (*)')
    if not fileName:
        return
    try:
        in_file = open(str(fileName), 'rb')
    except IOError:
        QtWidgets.QMessageBox.information(self, 'Unable to open file', 'There was an error opening "%s"' % fileName)
        return
    self.contacts = pickle.load(in_file)
    in_file.close()
    if len(self.contacts) == 0:
        QtWidgets.QMessageBox.information(self, 'No contacts in file', 'The file you are attempting to open contains no contacts.')
    else:
        for name, address in self.contacts:
            self.nameLine.setText(name)
            self.addressText.setText(address)
    self.updateInterface(self.NavigationMode)