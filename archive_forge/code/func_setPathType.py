from PySide2 import QtCore, QtGui, QtWidgets
import easing_rc
from ui_form import Ui_Form
def setPathType(self, pathType):
    self.m_pathType = pathType
    self.m_path = QtGui.QPainterPath()