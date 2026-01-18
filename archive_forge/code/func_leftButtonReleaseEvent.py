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
def leftButtonReleaseEvent(self, obj, event):
    ren = obj.GetCurrentRenderer()
    iren = ren.GetRenderWindow().GetInteractor()
    if self.mouse_motion == 0:
        pos = iren.GetEventPosition()
        iren.GetPicker().Pick(pos[0], pos[1], 0, ren)
    self.OnLeftButtonUp()