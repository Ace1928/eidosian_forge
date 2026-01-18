from __future__ import annotations
import itertools
import math
import os
import subprocess
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from pymatgen.core import PeriodicSite, Species, Structure
from pymatgen.util.coord import in_coord_list
def rotate_view(self, axis_ind=0, angle=0):
    """
        Rotate the camera view.

        Args:
            axis_ind: Index of axis to rotate. Defaults to 0, i.e., a-axis.
            angle: Angle to rotate by. Defaults to 0.
        """
    camera = self.ren.GetActiveCamera()
    if axis_ind == 0:
        camera.Roll(angle)
    elif axis_ind == 1:
        camera.Azimuth(angle)
    else:
        camera.Pitch(angle)
    self.ren_win.Render()