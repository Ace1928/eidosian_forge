from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
def create_datalist_example():
    return [('str', 'this is a string'), ('list', [0, '1', '3', '4']), ('list2', ['--', ('none', 'None'), ('--', 'Dashed'), ('-.', 'DashDot'), ('-', 'Solid'), ('steps', 'Steps'), (':', 'Dotted')]), ('float', 1.2), (None, 'Other:'), ('int', 12), ('font', ('Arial', 10, False, True)), ('color', '#123409'), ('bool', True), ('date', datetime.date(2010, 10, 10)), ('datetime', datetime.datetime(2010, 10, 10))]