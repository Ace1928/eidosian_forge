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
def get_complete_ph_dos(partial_dos_path, phonopy_yaml_path):
    """
    Creates a pymatgen CompletePhononDos from a partial_dos.dat and
    phonopy.yaml files.
    The second is produced when generating a Dos and is needed to extract
    the structure.

    Args:
        partial_dos_path: path to the partial_dos.dat file.
        phonopy_yaml_path: path to the phonopy.yaml file.
    """
    arr = np.loadtxt(partial_dos_path).transpose()
    dct = loadfn(phonopy_yaml_path)
    structure = get_structure_from_dict(dct['primitive_cell'])
    total_dos = PhononDos(arr[0], arr[1:].sum(axis=0))
    partial_doses = {}
    for site, p_dos in zip(structure, arr[1:]):
        partial_doses[site] = p_dos.tolist()
    return CompletePhononDos(structure, total_dos, partial_doses)