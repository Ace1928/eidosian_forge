import random
from PySide2 import QtCore, QtGui, QtWidgets, QtNetwork
def sendFortune(self):
    block = QtCore.QByteArray()
    out = QtCore.QDataStream(block, QtCore.QIODevice.WriteOnly)
    out.setVersion(QtCore.QDataStream.Qt_4_0)
    out.writeUInt16(0)
    fortune = self.fortunes[random.randint(0, len(self.fortunes) - 1)]
    out.writeString(fortune)
    out.device().seek(0)
    out.writeUInt16(block.size() - 2)
    clientConnection = self.tcpServer.nextPendingConnection()
    clientConnection.disconnected.connect(clientConnection.deleteLater)
    clientConnection.write(block)
    clientConnection.disconnectFromHost()