import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def set_default_hessian(self):
    n = len(self.atoms) * 3
    hessian = np.zeros((n, n))
    for i in range(n):
        hessian[i][i] = self.diagonal
    self.set_hessian(hessian)