import numpy as np
from ...Qt import QtCore, QtWidgets
from ...WidgetGroup import WidgetGroup
from ...widgets.ColorButton import ColorButton
from ...widgets.SpinBox import SpinBox
from ..Node import Node
def hideRow(self, name):
    w = self.ctrls[name]
    l = self.ui.layout().labelForField(w)
    w.hide()
    l.hide()