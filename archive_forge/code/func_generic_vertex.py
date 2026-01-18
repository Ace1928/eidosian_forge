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
def generic_vertex(self, vertex):
    if vertex in [v for v in self.Vertices if v is not vertex]:
        return False
    for arrow in self.Arrows:
        if arrow.too_close(vertex, tolerance=Arrow.epsilon + 2):
            return False
    return True