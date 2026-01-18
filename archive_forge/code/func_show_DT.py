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
def show_DT(self):
    """
        Display the DT hit counters next to each crossing.  Crossings
        that need to be flipped for the planar embedding have an
        asterisk.
        """
    for crossing in self.Crossings:
        crossing.locate()
        yshift = 0
        for arrow in (crossing.over, crossing.under):
            arrow.vectorize()
            if abs(arrow.dy) < 0.3 * abs(arrow.dx):
                yshift = 8
        flip = ' *' if crossing.flipped else ''
        self.DTlabels.append(self.canvas.create_text((crossing.x - 10, crossing.y - yshift), anchor=Tk_.E, text=str(crossing.hit1)))
        self.DTlabels.append(self.canvas.create_text((crossing.x + 10, crossing.y - yshift), anchor=Tk_.W, text=str(crossing.hit2) + flip))