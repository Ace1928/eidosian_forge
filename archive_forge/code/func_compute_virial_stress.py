import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import compare_atoms
from . import kimpy_wrappers
from . import neighborlist
@staticmethod
def compute_virial_stress(forces, coords, volume):
    """Compute the virial stress in Voigt notation.

        Parameters
        ----------
        forces : 2D array
            Partial forces on all atoms (padding included)

        coords : 2D array
            Coordinates of all atoms (padding included)

        volume : float
            Volume of cell

        Returns
        -------
        stress : 1D array
            stress in Voigt order (xx, yy, zz, yz, xz, xy)
        """
    stress = np.zeros(6)
    stress[0] = -np.dot(forces[:, 0], coords[:, 0]) / volume
    stress[1] = -np.dot(forces[:, 1], coords[:, 1]) / volume
    stress[2] = -np.dot(forces[:, 2], coords[:, 2]) / volume
    stress[3] = -np.dot(forces[:, 1], coords[:, 2]) / volume
    stress[4] = -np.dot(forces[:, 0], coords[:, 2]) / volume
    stress[5] = -np.dot(forces[:, 0], coords[:, 1]) / volume
    return stress