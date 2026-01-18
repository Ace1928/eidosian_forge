import warnings
import numpy as np
from ase.constraints import FixConstraint
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.utils import atoms_to_spglib_cell
def symmetrize_rank2(lattice, lattice_inv, stress_3_3, rot):
    """
    Return symmetrized stress

    lattice vectors expected as row vectors (same as ASE get_cell() convention),
    inv_lattice is its matrix inverse (reciprocal().T)
    """
    scaled_stress = np.dot(np.dot(lattice, stress_3_3), lattice.T)
    symmetrized_scaled_stress = np.zeros((3, 3))
    for r in rot:
        symmetrized_scaled_stress += np.dot(np.dot(r.T, scaled_stress), r)
    symmetrized_scaled_stress /= len(rot)
    sym = np.dot(np.dot(lattice_inv, symmetrized_scaled_stress), lattice_inv.T)
    return sym