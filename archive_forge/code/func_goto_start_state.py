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
def goto_start_state(self):
    self.canvas.delete('lock_error')
    self.canvas.delete(self.LiveArrow1)
    self.LiveArrow1 = None
    self.canvas.delete(self.LiveArrow2)
    self.LiveArrow2 = None
    self.ActiveVertex = None
    self.update_crosspoints()
    self.state = 'start_state'
    self.set_style()
    self.update_info()
    self.canvas.config(cursor='')