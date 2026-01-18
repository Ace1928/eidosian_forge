import sys
import signal
import time
from timeit import default_timer as clock
import wx
def ignore_keyboardinterrupts(func):
    """Decorator which causes KeyboardInterrupt exceptions to be ignored during
    execution of the decorated function.

    This is used by the inputhook functions to handle the event where the user
    presses CTRL+C while IPython is idle, and the inputhook loop is running. In
    this case, we want to ignore interrupts.
    """

    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            pass
    return wrapper