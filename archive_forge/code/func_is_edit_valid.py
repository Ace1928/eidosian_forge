from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
def is_edit_valid(edit):
    text = edit.text()
    state = edit.validator().validate(text, 0)[0]
    return state == QtGui.QDoubleValidator.State.Acceptable