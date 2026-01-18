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
def full_redraw(self):
    """
        Recolors and redraws all components, in DT order, and displays
        the legend linking colors to cusp indices.
        """
    components = self.arrow_components(include_isolated_vertices=True)
    self.colors = []
    for key in self.color_keys:
        self.canvas.delete(key)
    self.color_keys = []
    x, y, n = (10, 5, 0)
    self.palette.reset()
    for component in components:
        color = self.palette.new()
        self.colors.append(color)
        component[0].start.color = color
        for arrow in component:
            arrow.color = color
            arrow.end.color = color
            arrow.draw(self.Crossings)
        if self.style_var.get() != 'smooth':
            self.color_keys.append(self.canvas.create_text(x, y, text=str(n), fill=color, anchor=Tk_.NW, font='Helvetica 16 bold'))
        x, n = (x + 16, n + 1)
    for vertex in self.Vertices:
        vertex.draw()
    self.update_smooth()