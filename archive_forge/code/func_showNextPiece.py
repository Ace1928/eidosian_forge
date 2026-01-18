import random
from PySide2 import QtCore, QtGui, QtWidgets
def showNextPiece(self):
    if self.nextPieceLabel is not None:
        return
    dx = self.nextPiece.maxX() - self.nextPiece.minX() + 1
    dy = self.nextPiece.maxY() - self.nextPiece.minY() + 1
    pixmap = QtGui.QPixmap(dx * self.squareWidth(), dy * self.squareHeight())
    painter = QtGui.QPainter(pixmap)
    painter.fillRect(pixmap.rect(), self.nextPieceLabel.palette().background())
    for int in range(4):
        x = self.nextPiece.x(i) - self.nextPiece.minX()
        y = self.nextPiece.y(i) - self.nextPiece.minY()
        self.drawSquare(painter, x * self.squareWidth(), y * self.squareHeight(), self.nextPiece.shape())
    self.nextPieceLabel.setPixmap(pixmap)