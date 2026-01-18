import numpy as np
import ray
import ray.experimental.array.remote as ra
@staticmethod
def compute_block_shape(index, shape):
    lower = DistArray.compute_block_lower(index, shape)
    upper = DistArray.compute_block_upper(index, shape)
    return [u - l for l, u in zip(lower, upper)]