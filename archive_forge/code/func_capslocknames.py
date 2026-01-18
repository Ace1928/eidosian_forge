from functools import wraps
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import (
@interactor.decorate()
@printResult
def capslocknames(a=5):
    return a