import sys
from PySide2 import QtCore, QtGui, QtWidgets
def errorMessage(self):
    self.errorMessageDialog.showMessage('This dialog shows and remembers error messages. If the checkbox is checked (as it is by default), the shown message will be shown again, but if the user unchecks the box the message will not appear again if QErrorMessage.showMessage() is called with the same message.')
    self.errorLabel.setText("If the box is unchecked, the message won't appear again.")