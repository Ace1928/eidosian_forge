from .gui import *
from .CyOpenGL import (HoroballScene, OpenGLOrthoWidget,
from plink.ipython_tools import IPythonTkRoot
import os
import sys
def set_ties(self):
    if self.nbhd is None:
        return
    if len(self.tie_vars) == self.nbhd.num_cusps():
        for n, var in enumerate(self.tie_vars):
            self.nbhd.set_tie(n, var.get())