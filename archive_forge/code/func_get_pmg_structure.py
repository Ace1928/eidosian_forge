from __future__ import annotations
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from scipy.interpolate import InterpolatedUnivariateSpline
from pymatgen.core import Lattice, Structure
from pymatgen.phonon.bandstructure import PhononBandStructure, PhononBandStructureSymmLine
from pymatgen.phonon.dos import CompletePhononDos, PhononDos
from pymatgen.phonon.gruneisen import GruneisenParameter, GruneisenPhononBandStructureSymmLine
from pymatgen.phonon.thermal_displacements import ThermalDisplacementMatrices
from pymatgen.symmetry.bandstructure import HighSymmKpath
@requires(Phonopy, 'phonopy not installed!')
def get_pmg_structure(phonopy_structure: PhonopyAtoms) -> Structure:
    """
    Convert a PhonopyAtoms object to pymatgen Structure object.

    Args:
        phonopy_structure (PhonopyAtoms): A phonopy structure object.
    """
    lattice = phonopy_structure.cell
    frac_coords = phonopy_structure.scaled_positions
    symbols = phonopy_structure.symbols
    magmoms = getattr(phonopy_structure, 'magnetic_moments', [0] * len(symbols))
    site_props = {'phonopy_masses': phonopy_structure.masses, 'magmom': magmoms}
    return Structure(lattice, symbols, frac_coords, site_properties=site_props)