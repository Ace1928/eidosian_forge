import sys
import queue
import functools
import threading
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.debug import threadName
def runInStack(func):
    x = 'inside runInStack(func)'
    runInStack2(func)
    return x