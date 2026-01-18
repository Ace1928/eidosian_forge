import random
from PySide2 import QtCore, QtGui, QtWidgets
def pieceDropped(self, dropHeight):
    for i in range(4):
        x = self.curX + self.curPiece.x(i)
        y = self.curY - self.curPiece.y(i)
        self.setShapeAt(x, y, self.curPiece.shape())
    self.numPiecesDropped += 1
    if self.numPiecesDropped % 25 == 0:
        self.level += 1
        self.timer.start(self.timeoutTime(), self)
        self.levelChanged.emit(self.level)
    self.score += dropHeight + 7
    self.scoreChanged.emit(self.score)
    self.removeFullLines()
    if not self.isWaitingAfterLine:
        self.newPiece()