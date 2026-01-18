from __future__ import annotations
import os
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.constants import physical_constants, speed_of_light
from scipy.integrate import simps
from scipy.interpolate import interp1d
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.util.due import Doi, due
def optics(path=''):
    """Helper function to calculate optical absorption coefficient."""
    dir_gap, indir_gap = get_dir_indir_gap(path)
    run = Vasprun(path, occu_tol=0.01)
    new_en, new_abs = absorption_coefficient(run.dielectric)
    return (np.array(new_en, dtype=np.float64), np.array(new_abs, dtype=np.float64), dir_gap, indir_gap)