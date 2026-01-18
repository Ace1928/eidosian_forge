from math import exp, pi, sin, sqrt, cos, acos
import numpy as np
from ase.data import atomic_numbers
def set_damping(self, damping):
    """ set B-factor for thermal damping """
    self.damping = damping