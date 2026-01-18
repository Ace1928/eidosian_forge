from PySide2 import QtCore, QtGui, QtWidgets, QtNetwork
def requestNewFortune(self):
    self.getFortuneButton.setEnabled(False)
    self.blockSize = 0
    self.tcpSocket.abort()
    self.tcpSocket.connectToHost(self.hostLineEdit.text(), int(self.portLineEdit.text()))