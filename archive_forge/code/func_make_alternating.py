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
def make_alternating(self):
    """
        Changes crossings to make the projection alternating.
        Requires that all components be closed.
        """
    try:
        crossing_components = self.crossing_components()
    except ValueError:
        tkMessageBox.showwarning('Error', 'Please close up all components first.')
        return
    need_flipping = set()
    for component in self.DT_code()[0]:
        need_flipping.update((c for c in component if c < 0))
    for crossing in self.Crossings:
        if crossing.hit2 in need_flipping or crossing.hit1 in need_flipping:
            crossing.reverse()
    self.clear_text()
    self.update_info()
    for arrow in self.Arrows:
        arrow.draw(self.Crossings)
    self.update_smooth()