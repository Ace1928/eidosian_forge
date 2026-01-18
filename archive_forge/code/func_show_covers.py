import sys
import os
from .gui import *
from .polyviewer import PolyhedronViewer
from .horoviewer import HoroballViewer
from .CyOpenGL import GetColor
from .app_menus import browser_menus
from . import app_menus
from .number import Number
from . import database
from .exceptions import SnapPeaFatalError
from plink import LinkViewer, LinkEditor
from plink.ipython_tools import IPythonTkRoot
from spherogram.links.orthogonal import OrthogonalLinkDiagram
def show_covers(self, *args):
    self.state = 'ready'
    self.browse.config(default='active')
    self.action.config(default='normal')
    self.covers.delete(*self.covers.get_children())
    degree = int(self.degree_var.get())
    if self.cyclic_var.get():
        self.cover_list = self.manifold.covers(degree, cover_type='cyclic')
    else:
        self.cover_list = self.manifold.covers(degree)
    for n, N in enumerate(self.cover_list):
        cusps = repr(N.num_cusps())
        homology = repr(N.homology())
        name = N.name()
        cover_type = N.cover_info()['type']
        self.covers.insert('', 'end', values=(n, cover_type, cusps, homology))