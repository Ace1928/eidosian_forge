import copy
import numbers
from math import cos, sin, pi
import numpy as np
import ase.units as units
from ase.atom import Atom
from ase.cell import Cell
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.data import atomic_masses, atomic_masses_common
from ase.geometry import (wrap_positions, find_mic, get_angles, get_distances,
from ase.symbols import Symbols, symbols2numbers
from ase.utils import deprecated
def get_chemical_formula(self, mode='hill', empirical=False):
    """Get the chemical formula as a string based on the chemical symbols.

        Parameters:

        mode: str
            There are four different modes available:

            'all': The list of chemical symbols are contracted to a string,
            e.g. ['C', 'H', 'H', 'H', 'O', 'H'] becomes 'CHHHOH'.

            'reduce': The same as 'all' where repeated elements are contracted
            to a single symbol and a number, e.g. 'CHHHOCHHH' is reduced to
            'CH3OCH3'.

            'hill': The list of chemical symbols are contracted to a string
            following the Hill notation (alphabetical order with C and H
            first), e.g. 'CHHHOCHHH' is reduced to 'C2H6O' and 'SOOHOHO' to
            'H2O4S'. This is default.

            'metal': The list of chemical symbols (alphabetical metals,
            and alphabetical non-metals)

        empirical, bool (optional, default=False)
            Divide the symbol counts by their greatest common divisor to yield
            an empirical formula. Only for mode `metal` and `hill`.
        """
    return self.symbols.get_chemical_formula(mode, empirical)