import numpy as np
from ase.md.md import MolecularDynamics
from ase.parallel import world
def get_temperature(self):
    return self.temperature