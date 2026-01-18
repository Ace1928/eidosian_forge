from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
def apply_test(data):
    print('data:', data)