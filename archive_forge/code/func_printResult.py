from functools import wraps
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import (
def printResult(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        LAST_RESULT.value = func(*args, **kwargs)
        QtWidgets.QMessageBox.information(QtWidgets.QApplication.activeWindow(), 'Function Run!', f'Func result: {LAST_RESULT.value}')
    return wrapper