import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def get_lambdas(self, b, Gbar):
    lamdas = np.zeros(len(b))
    D = -Gbar / b
    absD = np.sqrt(np.dot(D, D))
    eps = 1e-12
    nminus = self.get_hessian_inertia(b)
    if absD < self.radius:
        if not self.transitionstate:
            self.write_log('Newton step')
            return lamdas
        elif nminus == 1:
            self.write_log('Newton step')
            return lamdas
        else:
            self.write_log('Wrong inertia of Hessian matrix: %2.2f %2.2f ' % (b[0], b[1]))
    else:
        self.write_log('Corrected Newton step: abs(D) = %2.2f ' % absD)
    if not self.transitionstate:
        upperlimit = min(0, b[0]) - eps
        lamda = find_lamda(upperlimit, Gbar, b, self.radius)
        lamdas += lamda
    else:
        upperlimit = min(-b[0], b[1], 0) - eps
        lamda = find_lamda(upperlimit, Gbar, b, self.radius)
        lamdas += lamda
        lamdas[0] -= 2 * lamda
    return lamdas