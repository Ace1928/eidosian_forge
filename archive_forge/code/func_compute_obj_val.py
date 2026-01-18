from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def compute_obj_val(self, x):
    obj_val = 0.5 * np.dot(x, self.work.data.P.dot(x)) + np.dot(self.work.data.q, x)
    if self.work.settings.scaling:
        obj_val *= self.work.scaling.cinv
    return obj_val