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
def save_figure(self, *args):
    directory = os.path.expanduser(mpl.rcParams['savefig.directory'])
    filename = _macosx.choose_save_file('Save the figure', directory, self.canvas.get_default_filename())
    if filename is None:
        return
    if mpl.rcParams['savefig.directory']:
        mpl.rcParams['savefig.directory'] = os.path.dirname(filename)
    self.canvas.figure.savefig(filename)