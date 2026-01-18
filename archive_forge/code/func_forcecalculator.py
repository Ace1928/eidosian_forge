import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def forcecalculator(self):
    return self.atoms.get_forces(md=True)