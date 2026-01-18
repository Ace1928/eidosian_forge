import sys
import re
from PySide2.QtCore import (QFile, Qt, QTextStream)
from PySide2.QtGui import (QColor, QFont, QKeySequence, QSyntaxHighlighter,
from PySide2.QtWidgets import (QAction, qApp, QApplication, QFileDialog, QMainWindow,
import syntaxhighlighter_rc
def setupFileMenu(self):
    fileMenu = self.menuBar().addMenu(self.tr('&File'))
    newFileAct = fileMenu.addAction(self.tr('&New...'))
    newFileAct.setShortcut(QKeySequence(QKeySequence.New))
    newFileAct.triggered.connect(self.newFile)
    openFileAct = fileMenu.addAction(self.tr('&Open...'))
    openFileAct.setShortcut(QKeySequence(QKeySequence.Open))
    openFileAct.triggered.connect(self.openFile)
    quitAct = fileMenu.addAction(self.tr('E&xit'))
    quitAct.setShortcut(QKeySequence(QKeySequence.Quit))
    quitAct.triggered.connect(self.close)
    helpMenu = self.menuBar().addMenu('&Help')
    helpMenu.addAction('About &Qt', qApp.aboutQt)