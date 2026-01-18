from functools import wraps
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import (
@interactor.decorate(runOptions=RunOptions.ON_ACTION)
@printResult
def runOnButton(a=10, b=20):
    return a + b