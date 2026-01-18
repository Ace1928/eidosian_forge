from gymnasium import core, spaces
import numpy as np
from ray.rllib.utils.annotations import PublicAPI
def extract_min_max(s):
    assert s.dtype == np.float64 or s.dtype == np.float32
    dim = np.int_(np.prod(s.shape))
    if type(s) == specs.Array:
        bound = np.inf * np.ones(dim, dtype=np.float32)
        return (-bound, bound)
    elif type(s) == specs.BoundedArray:
        zeros = np.zeros(dim, dtype=np.float32)
        return (s.minimum + zeros, s.maximum + zeros)