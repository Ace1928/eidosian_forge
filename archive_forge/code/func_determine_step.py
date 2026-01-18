import warnings
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def determine_step(self, dr, steplengths):
    """Old BFGS behaviour for scaling step lengths

        This keeps the behaviour of truncating individual steps. Some might
        depend of this as some absurd kind of stimulated annealing to find the
        global minimum.
        """
    dr /= np.maximum(steplengths / self.maxstep, 1.0).reshape(-1, 1)
    return dr