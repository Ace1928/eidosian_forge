import numpy as np
import ray
import ray.experimental.array.remote as ra
@staticmethod
def compute_block_lower(index, shape):
    if len(index) != len(shape):
        raise Exception('The fields `index` and `shape` must have the same length, but `index` is {} and `shape` is {}.'.format(index, shape))
    return [elem * BLOCK_SIZE for elem in index]