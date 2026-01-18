import os
import sys
import runpy
import types
from . import get_start_method, set_start_method
from . import process
from .context import reduction
from . import util
def is_forking(argv):
    """
    Return whether commandline indicates we are forking
    """
    if len(argv) >= 2 and argv[1] == '--multiprocessing-fork':
        return True
    else:
        return False