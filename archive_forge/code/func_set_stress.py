import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def set_stress(self, stress):
    """Set the applied stress.

        Must be a symmetric 3x3 tensor, a 6-vector representing a symmetric
        3x3 tensor, or a number representing the pressure.

        Use with care, it is better to set the correct stress when creating
        the object.
        """
    if np.isscalar(stress):
        stress = np.array([-stress, -stress, -stress, 0.0, 0.0, 0.0])
    else:
        stress = np.array(stress)
        if stress.shape == (3, 3):
            if not self._issymmetric(stress):
                raise ValueError('The external stress must be a symmetric tensor.')
            stress = np.array((stress[0, 0], stress[1, 1], stress[2, 2], stress[1, 2], stress[0, 2], stress[0, 1]))
        elif stress.shape != (6,):
            raise ValueError('The external stress has the wrong shape.')
    self.externalstress = stress