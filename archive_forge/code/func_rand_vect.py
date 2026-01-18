import numpy as np
from ase.optimize.optimize import Dynamics
def rand_vect(self):
    """Returns a random (Natoms,3) vector"""
    vect = self.rng.rand(len(self.atoms), 3) - 0.5
    return vect