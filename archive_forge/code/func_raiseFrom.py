import sys
import queue
import functools
import threading
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.debug import threadName
def raiseFrom():
    """Raise an exception from another
    """
    x = 'inside raiseFrom()'
    try:
        raiseException()
    except Exception as exc:
        raise Exception(f'Raised-from during exception handling {x} in {threadName()}') from exc