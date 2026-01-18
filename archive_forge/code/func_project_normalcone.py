from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def project_normalcone(self, z, y):
    tmp = z + y
    z = np.minimum(np.maximum(tmp, self.work.data.l), self.work.data.u)
    y = tmp - z
    return (z, y)