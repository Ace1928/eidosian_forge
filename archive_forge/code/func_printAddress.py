from PySide2.QtCore import (Qt, Signal)
from PySide2.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout)
from adddialogwidget import AddDialogWidget
def printAddress(name, address):
    print('Name:' + name)
    print('Address:' + address)