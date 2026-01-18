import numpy as np
import math
import inspect
import scipy.optimize
from scipy._lib._util import check_random_state
def one_cycle(self):
    """Do one cycle of the basinhopping algorithm
        """
    self.nstep += 1
    new_global_min = False
    accept, minres = self._monte_carlo_step()
    if accept:
        self.energy = minres.fun
        self.x = np.copy(minres.x)
        self.incumbent_minres = minres
        new_global_min = self.storage.update(minres)
    if self.disp:
        self.print_report(minres.fun, accept)
        if new_global_min:
            print('found new global minimum on step %d with function value %g' % (self.nstep, self.energy))
    self.xtrial = minres.x
    self.energy_trial = minres.fun
    self.accept = accept
    return new_global_min