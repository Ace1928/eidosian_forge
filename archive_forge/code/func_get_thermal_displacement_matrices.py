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
def get_thermal_displacement_matrices(thermal_displacements_yaml='thermal_displacement_matrices.yaml', structure_path='POSCAR'):
    """
    Function to read "thermal_displacement_matrices.yaml" from phonopy and return a list of
    ThermalDisplacementMatrices objects
    Args:
        thermal_displacements_yaml: path to thermal_displacement_matrices.yaml
        structure_path: path to POSCAR.

    Returns:
    """
    thermal_displacements_dict = loadfn(thermal_displacements_yaml)
    structure = Structure.from_file(structure_path)
    thermal_displacement_objects_list = []
    for matrix in thermal_displacements_dict['thermal_displacement_matrices']:
        thermal_displacement_objects_list.append(ThermalDisplacementMatrices(thermal_displacement_matrix_cart=matrix['displacement_matrices'], temperature=matrix['temperature'], structure=structure, thermal_displacement_matrix_cif=matrix['displacement_matrices_cif']))
    return thermal_displacement_objects_list