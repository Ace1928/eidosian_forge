import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
def sub_warning(msg, *args):
    if _logger:
        _logger.log(SUBWARNING, msg, *args)