import numpy as np
import ray
import ray.experimental.array.remote as ra
from . import core
@ray.remote
def qr_helper1(a_rc, y_ri, t, W_c):
    return a_rc - np.dot(y_ri, np.dot(t.T, W_c))