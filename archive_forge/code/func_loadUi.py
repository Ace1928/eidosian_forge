from PyQt5 import QtCore, QtGui, QtWidgets
from ..uiparser import UIParser
from .qobjectcreator import LoaderCreatorPolicy
def loadUi(self, filename, toplevelInst, resource_suffix):
    self.toplevelInst = toplevelInst
    return self.parse(filename, resource_suffix)