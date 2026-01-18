from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import (
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
def button_press_handler(event, canvas=None, toolbar=None):
    """
    The default Matplotlib button actions for extra mouse buttons.

    Parameters are as for `key_press_handler`, except that *event* is a
    `MouseEvent`.
    """
    if canvas is None:
        canvas = event.canvas
    if toolbar is None:
        toolbar = canvas.toolbar
    if toolbar is not None:
        button_name = str(MouseButton(event.button))
        if button_name in rcParams['keymap.back']:
            toolbar.back()
        elif button_name in rcParams['keymap.forward']:
            toolbar.forward()