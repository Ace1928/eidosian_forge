import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def set_hessian(self, hessian):
    self.hessian = hessian