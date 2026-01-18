import sys
import queue
import functools
import threading
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.debug import threadName
def raiseNested():
    """Raise an exception while handling another
    """
    x = 'inside raiseNested()'
    try:
        raiseException()
    except Exception:
        raise Exception(f'Raised during exception handling {x} in {threadName()}')