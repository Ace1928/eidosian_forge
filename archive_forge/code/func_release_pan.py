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
def release_pan(self, event):
    """Callback for mouse button release in pan/zoom mode."""
    if self._pan_info is None:
        return
    self.canvas.mpl_disconnect(self._pan_info.cid)
    self._id_drag = self.canvas.mpl_connect('motion_notify_event', self.mouse_move)
    for ax in self._pan_info.axes:
        ax.end_pan()
    self.canvas.draw_idle()
    self._pan_info = None
    self.push_current()