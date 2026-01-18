from functools import wraps
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import (
@interactor.decorate(a=10)
@printResult
def requiredParam(a, b=10):
    return a + b