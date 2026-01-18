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
def get_dir_indir_gap(run=''):
    """Get direct and indirect bandgaps for a vasprun.xml."""
    vasp_run = Vasprun(run)
    bandstructure = vasp_run.get_band_structure()
    dir_gap = bandstructure.get_direct_band_gap()
    indir_gap = bandstructure.get_band_gap()['energy']
    return (dir_gap, indir_gap)