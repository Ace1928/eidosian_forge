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
def showRegionOnScreen(region, outlineColor='red', filename='_showRegionOnScreen.png'):
    """
    TODO
    """
    screenshotIm = screenshot()
    draw = ImageDraw.Draw(screenshotIm)
    region = (region[0], region[1], region[2] + region[0], region[3] + region[1])
    draw.rectangle(region, outline=outlineColor)
    screenshotIm.save(filename)