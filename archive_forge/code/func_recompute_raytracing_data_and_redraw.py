from .hyperboloid_utilities import *
from .ideal_raytracing_data import *
from .finite_raytracing_data import *
from .hyperboloid_navigation import *
from .geodesics import Geodesics
from . import shaders
from snappy.CyOpenGL import SimpleImageShaderWidget
from snappy.SnapPy import vector, matrix
import math
def recompute_raytracing_data_and_redraw(self):
    self._initialize_raytracing_data()
    self.fix_view_state()
    self.redraw_if_initialized()