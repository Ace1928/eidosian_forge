import numpy as np
import ray
import ray.experimental.array.remote as ra
from . import core
@ray.remote
def qr_helper2(y_ri, a_rc):
    return np.dot(y_ri.T, a_rc)