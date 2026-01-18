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
def goto_drawing_state(self, x1, y1):
    self.ActiveVertex.expose()
    self.ActiveVertex.draw()
    x0, y0 = self.ActiveVertex.point()
    self.LiveArrow1 = self.canvas.create_line(x0, y0, x1, y1, fill='red')
    self.state = 'drawing_state'
    self.canvas.config(cursor='pencil')
    self.hide_DT()
    self.hide_labels()
    self.clear_text()