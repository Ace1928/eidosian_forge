from PySide2.QtCore import (Signal, QDataStream, QMutex, QMutexLocker,
from PySide2.QtGui import QIntValidator
from PySide2.QtWidgets import (QApplication, QDialogButtonBox, QGridLayout,
from PySide2.QtNetwork import (QAbstractSocket, QHostAddress, QNetworkInterface,
def showFortune(self, nextFortune):
    if nextFortune == self.currentFortune:
        self.requestNewFortune()
        return
    self.currentFortune = nextFortune
    self.statusLabel.setText(self.currentFortune)
    self.getFortuneButton.setEnabled(True)