from PySide2 import QtCore, QtGui, QtWidgets, QtNetwork
def displayError(self, socketError):
    if socketError == QtNetwork.QAbstractSocket.RemoteHostClosedError:
        pass
    elif socketError == QtNetwork.QAbstractSocket.HostNotFoundError:
        QtWidgets.QMessageBox.information(self, 'Fortune Client', 'The host was not found. Please check the host name and port settings.')
    elif socketError == QtNetwork.QAbstractSocket.ConnectionRefusedError:
        QtWidgets.QMessageBox.information(self, 'Fortune Client', 'The connection was refused by the peer. Make sure the fortune server is running, and check that the host name and port settings are correct.')
    else:
        QtWidgets.QMessageBox.information(self, 'Fortune Client', 'The following error occurred: %s.' % self.tcpSocket.errorString())
    self.getFortuneButton.setEnabled(True)