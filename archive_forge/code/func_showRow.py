import numpy as np
from ...Qt import QtCore, QtWidgets
from ...WidgetGroup import WidgetGroup
from ...widgets.ColorButton import ColorButton
from ...widgets.SpinBox import SpinBox
from ..Node import Node
def showRow(self, name):
    w = self.ctrls[name]
    l = self.ui.layout().labelForField(w)
    w.show()
    l.show()