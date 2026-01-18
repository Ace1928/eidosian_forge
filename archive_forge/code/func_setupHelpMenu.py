from PySide2 import QtCore, QtGui, QtWidgets
def setupHelpMenu(self):
    helpMenu = QtWidgets.QMenu('&Help', self)
    self.menuBar().addMenu(helpMenu)
    helpMenu.addAction('&About', self.about)
    helpMenu.addAction('About &Qt', QtWidgets.qApp.aboutQt)