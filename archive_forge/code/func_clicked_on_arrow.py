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
def clicked_on_arrow(self, vertex):
    for arrow in self.Arrows:
        if arrow.too_close(vertex):
            arrow.end.reverse_path(self.Crossings)
            self.update_info()
            return True
    return False