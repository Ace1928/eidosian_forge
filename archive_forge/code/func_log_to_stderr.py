import os
import sys
import threading
from . import process
from . import reduction
def log_to_stderr(self, level=None):
    """Turn on logging and add a handler which prints to stderr"""
    from .util import log_to_stderr
    return log_to_stderr(level)