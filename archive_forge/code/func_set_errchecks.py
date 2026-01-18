import atexit
import struct
import warnings
import pyglet
from . import com
from . import constants
from .types import *
def set_errchecks(lib):
    """Set errcheck hook on all functions we have defined."""
    for key in lib.__dict__:
        if key.startswith('_'):
            continue
        lib.__dict__[key].errcheck = win32_errcheck