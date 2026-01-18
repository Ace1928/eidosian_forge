import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
def is_exiting():
    """
    Returns true if the process is shutting down
    """
    return _exiting or _exiting is None