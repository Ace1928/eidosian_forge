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
def mouse_moved(self, event):
    """
        Handler for mouse motion events.
        """
    if self.style_var.get() == 'smooth':
        return
    canvas = self.canvas
    X, Y = (event.x, event.y)
    x, y = (canvas.canvasx(X), canvas.canvasy(Y))
    self.cursorx, self.cursory = (X, Y)
    if self.state == 'start_state':
        self.set_start_cursor(x, y)
    elif self.state == 'drawing_state':
        x0, y0, x1, y1 = self.canvas.coords(self.LiveArrow1)
        self.canvas.coords(self.LiveArrow1, x0, y0, x, y)
    elif self.state == 'dragging_state':
        if self.shifting:
            self.window.event_generate('<Return>')
            return 'break'
        else:
            self.move_active(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))