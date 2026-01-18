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
def start_event_loop(self, timeout=0):
    with _maybe_allow_interrupt():
        self._start_event_loop(timeout=timeout)