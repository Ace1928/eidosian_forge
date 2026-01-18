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
def compute_pi_one(self):
    fun_gp = self.manifold.fundamental_group(simplify_presentation=self.simplify_var.get(), minimize_number_of_generators=self.minimize_var.get(), fillings_may_affect_generators=self.gens_change_var.get())
    self.pi_one.set(repr(fun_gp))