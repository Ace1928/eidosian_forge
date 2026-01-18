import time
import warnings
from math import sqrt
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.constraints import UnitCellFilter
from ase.utils.linesearch import LineSearch
from ase.utils.linesearcharmijo import LineSearchArmijo
from ase.optimize.precon.precon import make_precon
def reset_hessian(self):
    """
        Throw away history of the Hessian
        """
    self._just_reset_hessian = True
    self.s = []
    self.y = []
    self.rho = []