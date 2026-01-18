import numpy as np
from ...Qt import QtCore, QtWidgets
from ...WidgetGroup import WidgetGroup
from ...widgets.ColorButton import ColorButton
from ...widgets.SpinBox import SpinBox
from ..Node import Node
def disconnectFromPlot(self, plot):
    """Define what happens when the node is disconnected from a plot"""
    raise Exception('Must be re-implemented in subclass')