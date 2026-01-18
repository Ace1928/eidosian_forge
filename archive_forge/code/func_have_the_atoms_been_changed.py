import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def have_the_atoms_been_changed(self):
    """Checks if the user has modified the positions or momenta of the atoms"""
    limit = 1e-10
    h = self._getbox()
    if max(abs((h - self.h).ravel())) > limit:
        self._warning('The computational box has been modified.')
        return 1
    expected_r = np.dot(self.q + 0.5, h)
    err = max(abs((expected_r - self.atoms.get_positions()).ravel()))
    if err > limit:
        self._warning('The atomic positions have been modified: ' + str(err))
        return 1
    return 0