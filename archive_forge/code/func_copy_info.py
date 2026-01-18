import os, time, webbrowser
from .gui import *
from . import smooth
from .vertex import Vertex
from .arrow import Arrow
from .crossings import Crossing, ECrossing
from .colors import Palette
from .dialog import InfoDialog
from .manager import LinkManager
from .viewer import LinkViewer
from .version import version
from .ipython_tools import IPythonTkRoot
def copy_info(self, event):
    self.window.clipboard_clear()
    if self.infotext.selection_present():
        self.window.clipboard_append(self.infotext.selection_get())
        self.infotext.selection_clear()