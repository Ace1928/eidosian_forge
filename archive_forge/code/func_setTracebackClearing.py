import sys
import threading
import time
import traceback
from types import SimpleNamespace
def setTracebackClearing(clear=True):
    """
    Enable or disable traceback clearing.
    By default, clearing is disabled and Python will indefinitely store unhandled exception stack traces.
    This function is provided since Python's default behavior can cause unexpected retention of 
    large memory-consuming objects.
    """
    global clear_tracebacks
    clear_tracebacks = clear