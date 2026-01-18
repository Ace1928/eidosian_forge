import numpy as np
from ase.optimize.optimize import Dynamics
def subtract_projection(a, b):
    """returns new vector that removes vector a's projection vector b. Is
    also equivalent to the vector rejection."""
    aout = a - np.vdot(a, b) / np.vdot(b, b) * b
    return aout