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
def end_dragging_state(self):
    if not self.verify_drag():
        raise ValueError
    if self.lock_var.get():
        self.detach_cursor()
        self.saved_crossing_data = None
    else:
        x, y = (float(self.cursorx), float(self.cursory))
        self.ActiveVertex.x, self.ActiveVertex.y = (x, y)
    endpoint = None
    if self.ActiveVertex.is_endpoint():
        other_ends = [v for v in self.Vertices if v.is_endpoint() and v is not self.ActiveVertex]
        if self.ActiveVertex in other_ends:
            endpoint = other_ends[other_ends.index(self.ActiveVertex)]
            self.ActiveVertex.swallow(endpoint, self.palette)
            self.Vertices = [v for v in self.Vertices if v is not endpoint]
        self.update_crossings(self.ActiveVertex.in_arrow)
        self.update_crossings(self.ActiveVertex.out_arrow)
    if endpoint is None and (not self.generic_vertex(self.ActiveVertex)):
        raise ValueError
    self.ActiveVertex.expose()
    if self.style_var.get() != 'smooth':
        if self.ActiveVertex.in_arrow:
            self.ActiveVertex.in_arrow.expose()
        if self.ActiveVertex.out_arrow:
            self.ActiveVertex.out_arrow.expose()
    self.goto_start_state()