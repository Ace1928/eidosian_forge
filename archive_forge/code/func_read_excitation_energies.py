from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as cst
from monty.io import zopen
from scipy.stats import norm
from pymatgen.core import Composition, Element, Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.coord import get_angle
from pymatgen.util.plotting import pretty_plot
def read_excitation_energies(self):
    """
        Read a excitation energies after a TD-DFT calculation.

        Returns:
            A list: A list of tuple for each transition such as
                    [(energie (eV), lambda (nm), oscillatory strength), ... ]
        """
    transitions = []
    with zopen(self.filename, mode='r') as file:
        line = file.readline()
        td = False
        while line != '':
            if re.search('^\\sExcitation energies and oscillator strengths:', line):
                td = True
            if td and re.search('^\\sExcited State\\s*\\d', line):
                val = [float(v) for v in float_patt.findall(line)]
                transitions.append(tuple(val[:3]))
            line = file.readline()
    return transitions