import time
from .gui import *
from .CyOpenGL import *
from .export_stl import stl
from . import filedialog
from plink.ipython_tools import IPythonTkRoot
def new_polyhedron(self, new_facedicts):
    self.empty = len(new_facedicts) == 0
    self.widget.tk.call(self.widget._w, 'makecurrent')
    try:
        self.polyhedron.delete_resource()
    except AttributeError:
        pass
    self.polyhedron = HyperbolicPolyhedron(new_facedicts, self.model_var, self.sphere_var, togl_widget=self.widget)
    self.widget.redraw_impl = self.polyhedron.draw
    self.widget.redraw_if_initialized()