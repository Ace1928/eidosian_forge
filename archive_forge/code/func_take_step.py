import numpy as np
import math
import inspect
import scipy.optimize
from scipy._lib._util import check_random_state
def take_step(self, x):
    self.nstep += 1
    self.nstep_tot += 1
    if self.nstep % self.interval == 0:
        self._adjust_step_size()
    return self.takestep(x)