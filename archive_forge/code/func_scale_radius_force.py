import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def scale_radius_force(f, r):
    scale = 1.0
    g = abs(f - 1)
    if g < 0.01:
        scale *= 1.4
    if g < 0.05:
        scale *= 1.4
    if g < 0.1:
        scale *= 1.4
    if g < 0.4:
        scale *= 1.4
    if g > 0.5:
        scale *= 1.0 / 1.4
    if g > 0.7:
        scale *= 1.0 / 1.4
    if g > 1.0:
        scale *= 1.0 / 1.4
    return scale