from typing import List, Optional, Union
import numpy as np
import tensorflow as tf
from .feature_extraction_utils import BatchFeature
from .tokenization_utils_base import BatchEncoding
from .utils import logging
def functional_layernorm(inputs, weight, bias, epsilon=1e-05, axis=-1):
    if weight.shape.rank != 1 or bias.shape.rank != 1 or (not isinstance(axis, int)):
        raise NotImplementedError('Only 1D weight and bias tensors are supported for now, with only a single axis.')
    mean, variance = tf.nn.moments(inputs, axes=[axis], keepdims=True)
    if axis != -1:
        shape = [1] * inputs.shape.rank
        shape[axis] = shape_list(inputs)[axis]
        weight = tf.reshape(weight, shape)
        bias = tf.reshape(bias, shape)
    outputs = tf.nn.batch_normalization(inputs, mean, variance, offset=bias, scale=weight, variance_epsilon=epsilon)
    return outputs