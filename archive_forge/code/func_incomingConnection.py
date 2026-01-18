import random
from PySide2.QtCore import (Signal, QByteArray, QDataStream, QIODevice,
from PySide2.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel,
from PySide2.QtNetwork import (QHostAddress, QNetworkInterface, QTcpServer,
def incomingConnection(self, socketDescriptor):
    fortune = self.fortunes[random.randint(0, len(self.fortunes) - 1)]
    thread = FortuneThread(socketDescriptor, fortune, self)
    thread.finished.connect(thread.deleteLater)
    thread.start()