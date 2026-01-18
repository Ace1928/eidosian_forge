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
def build_inside_view(self):
    if not self.manifold.is_orientable():
        text = 'Inside view for non-orientable manifolds such as %s is not supported yet.' % self.manifold.name()
        return ttk.Label(self, text=text)
    try:
        from .raytracing.inside_viewer import InsideViewer
        self.inside_view = InsideViewer(self, self.manifold, fillings_changed_callback=self.update_modeline_and_side_panel)
        self.fillings_changed_callback = self.inside_view.pull_fillings_from_manifold
        return self.inside_view
    except Exception:
        import traceback
        text = 'Could not instantiate inside view. Error was:\n\n%s' % traceback.format_exc()
        return ttk.Label(self, text=text)