import contextlib
import os
import signal
import socket
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib._pylab_helpers import Gcf
from . import _macosx
from .backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import (
def show(self):
    if not self._shown:
        self._show()
        self._shown = True
    if mpl.rcParams['figure.raise_window']:
        self._raise()