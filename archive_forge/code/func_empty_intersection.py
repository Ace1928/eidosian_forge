import functools
import tensorflow as tf
def empty_intersection():
    return (intersection_indices, empty_tensor((0,) + x1.values.shape[1:], x1.dtype), empty_tensor((0,) + x2.values.shape[1:], x2.dtype))