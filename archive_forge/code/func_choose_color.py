from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
def choose_color(self):
    color = QtWidgets.QColorDialog.getColor(self._color, self.parentWidget(), '', QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel)
    if color.isValid():
        self.set_color(color)