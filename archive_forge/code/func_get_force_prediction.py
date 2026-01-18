import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def get_force_prediction(self, G):
    Gbar = np.dot(G, np.transpose(self.V))
    dGbar_actual = Gbar - self.old_gbar
    dGbar_predicted = Gbar - self.gbar_estimate
    f = np.dot(dGbar_actual, dGbar_predicted) / np.dot(dGbar_actual, dGbar_actual)
    self.write_log('Force prediction factor ' + str(f))
    return f