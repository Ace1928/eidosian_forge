from .gui import *
from .CyOpenGL import (HoroballScene, OpenGLOrthoWidget,
from plink.ipython_tools import IPythonTkRoot
import os
import sys
def start_radius(self, event):
    self.cusp_moving = True
    self.moving_cusp = index = event.widget.index
    self.last_slider_value = self.cusp_sliders[index].get()
    self.update_radius()