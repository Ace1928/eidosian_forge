from PySide2 import QtCore, QtGui, QtWidgets, QtNetwork
def readFortune(self):
    instr = QtCore.QDataStream(self.tcpSocket)
    instr.setVersion(QtCore.QDataStream.Qt_4_0)
    if self.blockSize == 0:
        if self.tcpSocket.bytesAvailable() < 2:
            return
        self.blockSize = instr.readUInt16()
    if self.tcpSocket.bytesAvailable() < self.blockSize:
        return
    nextFortune = instr.readString()
    try:
        nextFortune = str(nextFortune, encoding='ascii')
    except TypeError:
        pass
    if nextFortune == self.currentFortune:
        QtCore.QTimer.singleShot(0, self.requestNewFortune)
        return
    self.currentFortune = nextFortune
    self.statusLabel.setText(self.currentFortune)
    self.getFortuneButton.setEnabled(True)