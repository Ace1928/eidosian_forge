import sys
import math
from PySide2 import QtCore, QtGui, QtWidgets
def createSlider(self, changedSignal, setterSlot):
    slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
    slider.setRange(0, 360 * 16)
    slider.setSingleStep(16)
    slider.setPageStep(15 * 16)
    slider.setTickInterval(15 * 16)
    slider.setTickPosition(QtWidgets.QSlider.TicksRight)
    self.glWidget.connect(slider, QtCore.SIGNAL('valueChanged(int)'), setterSlot)
    self.connect(self.glWidget, changedSignal, slider, QtCore.SLOT('setValue(int)'))
    return slider