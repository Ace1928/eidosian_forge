from .gui import *
from .CyOpenGL import (HoroballScene, OpenGLOrthoWidget,
from plink.ipython_tools import IPythonTkRoot
import os
import sys
def set_eye(self):
    self.which_cusp = self.eye_var.get()
    self.rebuild()