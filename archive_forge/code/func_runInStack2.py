import sys
import queue
import functools
import threading
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.debug import threadName
def runInStack2(func):
    x = 'inside runInStack2(func)'
    runInStack3(func)
    return x