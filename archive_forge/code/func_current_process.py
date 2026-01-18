import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
def current_process():
    """
    Return process object representing the current process
    """
    return _current_process