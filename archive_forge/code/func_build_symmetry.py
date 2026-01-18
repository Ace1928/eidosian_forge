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
def build_symmetry(self):
    style = self.style
    frame = ttk.Frame(self)
    frame.grid_columnconfigure(0, weight=1)
    self.symmetry = SelectableText(frame, labeltext='Symmetry Group:', width=30, depth=1)
    self.symmetry.grid(row=0, column=0, pady=20)
    message1 = ttk.Label(frame, text='Future releases of SnapPy will show more information here.')
    message2 = ttk.Label(frame, text='Type SymmetryGroup.<tab> in the command shell to see what is available.')
    message1.grid(row=1, column=0, pady=(40, 10))
    message2.grid(row=2, column=0)
    return frame