import numpy as np
import warnings
from scipy.optimize import minimize
from ase.parallel import world
from ase.io.jsonio import write_json
from ase.optimize.optimize import Optimizer
from ase.optimize.gpmin.gp import GaussianProcess
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.prior import ConstantPrior
def fit_to_batch(self):
    """Fit hyperparameters keeping the ratio noise/weight fixed"""
    ratio = self.noise / self.kernel.weight
    self.fit_hyperparameters(np.array(self.x_list), np.array(self.y_list), eps=self.eps)
    self.noise = ratio * self.kernel.weight