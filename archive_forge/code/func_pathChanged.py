from PySide2 import QtCore, QtGui, QtWidgets
import easing_rc
from ui_form import Ui_Form
def pathChanged(self, index):
    self.m_anim.setPathType(index)