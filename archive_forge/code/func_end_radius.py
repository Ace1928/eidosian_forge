from .gui import *
from .CyOpenGL import (HoroballScene, OpenGLOrthoWidget,
from plink.ipython_tools import IPythonTkRoot
import os
import sys
def end_radius(self, event):
    self.cusp_moving = False
    self.rebuild()