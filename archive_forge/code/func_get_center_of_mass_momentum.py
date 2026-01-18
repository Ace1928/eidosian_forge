import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def get_center_of_mass_momentum(self):
    """Get the center of mass momentum."""
    return self.atoms.get_momenta().sum(0)