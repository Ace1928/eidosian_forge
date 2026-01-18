import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def stresscalculator(self):
    return self.atoms.get_stress(include_ideal_gas=True)