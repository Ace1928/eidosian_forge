import time
from .gui import *
from .CyOpenGL import *
from .export_stl import stl
from . import filedialog
from plink.ipython_tools import IPythonTkRoot
def new_model(self):
    self.widget.redraw_if_initialized()