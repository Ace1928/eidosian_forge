import numpy as np
import ray
@ray.remote
def det(a):
    return np.linalg.det(a)