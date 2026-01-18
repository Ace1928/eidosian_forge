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
def crossed_arrows(self, arrow, ignore_list=[]):
    """
        Return a tuple containing the arrows of the diagram which are
        crossed by the given arrow, in order along the given arrow.
        """
    if arrow is None:
        return tuple()
    arrow.vectorize()
    crosslist = []
    for n, diagram_arrow in enumerate(self.Arrows):
        if arrow == diagram_arrow or diagram_arrow in ignore_list:
            continue
        t = arrow ^ diagram_arrow
        if t is not None:
            crosslist.append((t, n))
    return tuple((n for _, n in sorted(crosslist)))