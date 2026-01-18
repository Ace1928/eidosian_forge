from PySide2 import QtCore, QtGui, QtWidgets
def submitContact(self):
    name = self.nameLine.text()
    address = self.addressText.toPlainText()
    if name == '' or address == '':
        QtWidgets.QMessageBox.information(self, 'Empty Field', 'Please enter a name and address.')
        return
    if self.currentMode == self.AddingMode:
        if name not in self.contacts:
            self.contacts[name] = address
            QtWidgets.QMessageBox.information(self, 'Add Successful', '"%s" has been added to your address book.' % name)
        else:
            QtWidgets.QMessageBox.information(self, 'Add Unsuccessful', 'Sorry, "%s" is already in your address book.' % name)
            return
    elif self.currentMode == self.EditingMode:
        if self.oldName != name:
            if name not in self.contacts:
                QtWidgets.QMessageBox.information(self, 'Edit Successful', '"%s" has been edited in your address book.' % self.oldName)
                del self.contacts[self.oldName]
                self.contacts[name] = address
            else:
                QtWidgets.QMessageBox.information(self, 'Edit Unsuccessful', 'Sorry, "%s" is already in your address book.' % name)
                return
        elif self.oldAddress != address:
            QtWidgets.QMessageBox.information(self, 'Edit Successful', '"%s" has been edited in your address book.' % name)
            self.contacts[name] = address
    self.updateInterface(self.NavigationMode)