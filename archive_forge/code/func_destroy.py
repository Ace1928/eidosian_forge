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
def destroy(self):
    while self.canvas._timers:
        timer = self.canvas._timers.pop()
        timer.stop()
    super().destroy()