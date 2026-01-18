from .gui import *
from .CyOpenGL import (HoroballScene, OpenGLOrthoWidget,
from plink.ipython_tools import IPythonTkRoot
import os
import sys
def configure_sliders(self):
    nbhd = self.nbhd
    if self.nbhd is None:
        return
    slider_width = 30
    size = 330 - slider_width
    max_reach = nbhd.max_reach()
    for n in range(nbhd.num_cusps()):
        stopper_color = self.cusp_colors[nbhd.stopper(n)]
        stop = float(nbhd.stopping_displacement(n))
        length = int(stop * size / max_reach) + slider_width
        disp = float(nbhd.get_displacement(n))
        position = 100.0 * disp / stop
        self.cusp_sliders[n].set(position)
        self.slider_frames[n].config(background=stopper_color)
        self.volume_labels[n].config(text='%.4f' % nbhd.volume(n))
        self.cusp_sliders[n].config(length=length, command=self.update_radius)
    self.update_idletasks()