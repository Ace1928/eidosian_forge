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
def locateCenterOnScreenNear(image, x, y, **kwargs):
    """
    TODO
    """
    coords = locateOnScreenNear(image, x, y, **kwargs)
    if coords is None:
        return None
    else:
        return center(coords)