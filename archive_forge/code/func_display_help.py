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
def display_help(self):
    """Display the help for various keyboard shortcuts."""
    helptxt = ['h : Toggle help', 'A/a, B/b or C/c : Increase/decrease cell by one a, b or c unit vector', '# : Toggle showing of polyhedrons', '-: Toggle showing of bonds', 'r : Reset camera direction', f'[/]: Decrease or increase poly_radii_tol_factor by 0.05. Value = {self.poly_radii_tol_factor}', 'Up/Down: Rotate view along Up direction by 90 clockwise/anticlockwise', 'Left/right: Rotate view along camera direction by 90 clockwise/anticlockwise', 's: Save view to image.png', 'o: Orthogonalize structure', 'n: Move to next structure', 'p: Move to previous structure', 'm: Animated movie of the structures']
    self.helptxt_mapper.SetInput('\n'.join(helptxt))
    self.helptxt_actor.SetPosition(10, 10)
    self.helptxt_actor.VisibilityOn()