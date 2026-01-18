from __future__ import annotations
import logging
import os
import re
import warnings
from glob import glob
from itertools import chain
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from monty.re import regrep
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.cp2k.inputs import Keyword
from pymatgen.io.cp2k.sets import Cp2kInput
from pymatgen.io.cp2k.utils import natural_keys, postprocessor
from pymatgen.io.xyz import XYZ
def parse_opt_steps(self):
    """Parse the geometry optimization information."""
    total_energy = re.compile('\\s+Total Energy\\s+=\\s+(-?\\d+.\\d+)')
    real_energy_change = re.compile('\\s+Real energy change\\s+=\\s+(-?\\d+.\\d+)')
    prediced_change_in_energy = re.compile('\\s+Predicted change in energy\\s+=\\s+(-?\\d+.\\d+)')
    scaling_factor = re.compile('\\s+Scaling factor\\s+=\\s+(-?\\d+.\\d+)')
    step_size = re.compile('\\s+Step size\\s+=\\s+(-?\\d+.\\d+)')
    trust_radius = re.compile('\\s+Trust radius\\s+=\\s+(-?\\d+.\\d+)')
    used_time = re.compile('\\s+Used time\\s+=\\s+(-?\\d+.\\d+)')
    pressure_deviation = re.compile('\\s+Pressure Deviation.*=\\s+(-?\\d+.\\d+)')
    pressure_tolerance = re.compile('\\s+Pressure Tolerance.*=\\s+(-?\\d+.\\d+)')
    self.read_pattern({'total_energy': total_energy, 'real_energy_change': real_energy_change, 'predicted_change_in_energy': prediced_change_in_energy, 'scaling_factor': scaling_factor, 'step_size': step_size, 'trust_radius': trust_radius, 'used_time': used_time, 'pressure_deviation': pressure_deviation, 'pressure_tolerance': pressure_tolerance}, terminate_on_match=False, postprocess=float)
    decrease_in_energy = re.compile('\\s+Decrease in energy\\s+=\\s+(\\w+)')
    converged_step_size = re.compile('\\s+Convergence in step size\\s+=\\s+(\\w+)')
    converged_rms_step = re.compile('\\s+Convergence in RMS step\\s+=\\s+(\\w+)')
    converged_in_grad = re.compile('\\s+Conv\\. in gradients\\s+=\\s+(\\w+)')
    converged_in_rms_grad = re.compile('\\s+Conv\\. in RMS gradients\\s+=\\s+(\\w+)')
    pressure_converged = re.compile('\\s+Conv\\. for  PRESSURE\\s+=\\s+(\\w+)')
    self.read_pattern({'decrease_in_energy': decrease_in_energy, 'converged_step_size': converged_step_size, 'converged_rms_step': converged_rms_step, 'converged_in_grad': converged_in_grad, 'converged_in_rms_grad': converged_in_rms_grad, 'pressure_converged': pressure_converged}, terminate_on_match=False, postprocess=postprocessor)