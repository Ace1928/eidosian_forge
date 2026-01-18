import inspect
import os
import sys
import traceback
import types
import tensorflow.compat.v2 as tf
def format_argument_value(value):
    if isinstance(value, tf.Tensor):
        return f'tf.Tensor(shape={value.shape}, dtype={value.dtype.name})'
    return repr(value)