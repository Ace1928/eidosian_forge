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
@classmethod
def draw_if_interactive(cls):
    manager_class = cls.FigureCanvas.manager_class
    backend_is_interactive = manager_class.start_main_loop != FigureManagerBase.start_main_loop or manager_class.pyplot_show != FigureManagerBase.pyplot_show
    if backend_is_interactive and is_interactive():
        manager = Gcf.get_active()
        if manager:
            manager.canvas.draw_idle()