import operator
import weakref
import numpy as np
from .. import functions as fn
from .. import colormap
from ..colormap import ColorMap
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorMapButton import ColorMapMenu
from .GraphicsWidget import GraphicsWidget
from .GradientPresets import Gradients
def setColorMode(self, cm):
    """
        Set the color mode for the gradient. Options are: 'hsv', 'rgb'
        
        """
    if cm not in ['rgb', 'hsv']:
        raise Exception("Unknown color mode %s. Options are 'rgb' and 'hsv'." % str(cm))
    try:
        self.rgbAction.blockSignals(True)
        self.hsvAction.blockSignals(True)
        self.rgbAction.setChecked(cm == 'rgb')
        self.hsvAction.setChecked(cm == 'hsv')
    finally:
        self.rgbAction.blockSignals(False)
        self.hsvAction.blockSignals(False)
    self.colorMode = cm
    self.sigTicksChanged.emit(self)
    self.sigGradientChangeFinished.emit(self)