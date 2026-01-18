from PySide2.QtWidgets import (QMainWindow, QAction, QFileDialog, QApplication)
from addresswidget import AddressWidget
def saveFile(self):
    filename, _ = QFileDialog.getSaveFileName(self)
    if filename:
        self.addressWidget.writeToFile(filename)