import numpy as np
from ase.md.md import MolecularDynamics
from ase.parallel import world
def set_taut(self, taut):
    self.taut = taut