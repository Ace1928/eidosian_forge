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
def press_zoom(self, event):
    """Callback for mouse button press in zoom to rect mode."""
    if event.button not in [MouseButton.LEFT, MouseButton.RIGHT] or event.x is None or event.y is None:
        return
    axes = [a for a in self.canvas.figure.get_axes() if a.in_axes(event) and a.get_navigate() and a.can_zoom()]
    if not axes:
        return
    if self._nav_stack() is None:
        self.push_current()
    id_zoom = self.canvas.mpl_connect('motion_notify_event', self.drag_zoom)
    if hasattr(axes[0], '_colorbar'):
        cbar = axes[0]._colorbar.orientation
    else:
        cbar = None
    self._zoom_info = self._ZoomInfo(direction='in' if event.button == 1 else 'out', start_xy=(event.x, event.y), axes=axes, cid=id_zoom, cbar=cbar)