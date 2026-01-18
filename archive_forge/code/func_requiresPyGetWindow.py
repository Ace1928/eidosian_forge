import collections
import datetime
import functools
import os
import subprocess
import sys
import time
import errno
from contextlib import contextmanager
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import __version__ as PIL__version__
from PIL import ImageGrab
def requiresPyGetWindow(wrappedFunction):
    """
    A decorator that marks a function as requiring PyGetWindow to be installed.
    This raises PyScreezeException if Pillow wasn't imported.
    """

    @functools.wraps(wrappedFunction)
    def wrapper(*args, **kwargs):
        if _PYGETWINDOW_UNAVAILABLE:
            raise PyScreezeException('The PyGetWindow package is required to use this function.')
        return wrappedFunction(*args, **kwargs)
    return wrapper