from functools import wraps
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import (
@interactor.decorate()
@printResult
def stringParams(a='5', b='6'):
    return a + b