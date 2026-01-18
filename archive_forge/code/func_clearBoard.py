import random
from PySide2 import QtCore, QtGui, QtWidgets
def clearBoard(self):
    self.board = [NoShape for i in range(TetrixBoard.BoardHeight * TetrixBoard.BoardWidth)]